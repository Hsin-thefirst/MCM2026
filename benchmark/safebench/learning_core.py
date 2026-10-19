"""Counterfactual transitions and finite-horizon fitted Q iteration without extra dependencies."""
import hashlib
import time
import numpy as np
from safeinvest.contracts import Action, Decision, validate_weights
from .scenario_utility import REVIEW_EXPLANATION

LEARNING_NAMES = ('fqi_student', 'fqi_teacher', 'fqi_portfolio')
FEATURE_NAMES = ('equity_weight', 'bond_weight', 'drawdown', 'maximum_drawdown',
    'last_equity_real', 'last_bond_real', 'last_bill_real', 'last_inflation',
    'equity_mean_3y', 'equity_volatility_3y', 'cost_rate', 'review_pending')
PORTFOLIO_FEATURES = (0, 1, 2, 3, 10, 11)


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def market_features(history):
    last = history[-1]
    real = [(1 + r) / (1 + last.inflation) - 1 for r in last.nominal_returns]
    equity = [(1 + o.nominal_returns[0]) / (1 + o.inflation) - 1 for o in history[-3:]]
    return (*real, last.inflation, float(np.mean(equity)), float(np.std(equity)))


def state_features(weights, drawdown, maximum_drawdown, history, cost, pending):
    return np.array([*weights[:2], drawdown, maximum_drawdown,
                     *market_features(history), cost, float(pending)], dtype=float)


def action_fees(features, config):
    x = np.atleast_2d(features)
    weights = np.column_stack((x[:, 0], x[:, 1], 1 - x[:, 0] - x[:, 1]))
    fees = np.zeros((len(x), 4))
    fees[:, 1:] = x[:, 10, None] * .5 * np.abs(
        weights[:, None, :] - np.array(config['learning_targets'])[None, :, :]).sum(axis=2)
    forced = (x[:, 11] > .5) | (x[:, 2] >= config['review_drawdown'])
    fees[forced] = 0
    return fees


def fit_tree(x, y, depth, min_leaf, candidates, features):
    """Deterministic multi-output CART using exact prefix sums at bounded split candidates."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Invalid tree training arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite tree training arrays')

    def grow(indices, level):
        values = y[indices]
        total = values.sum(axis=0)
        node = dict(value=(total / len(indices)).tolist(), n=len(indices))
        if level == depth or len(indices) < 2 * min_leaf:
            return node
        best = None
        parent = float(total @ total / len(indices))
        for feature in features:
            order = np.argsort(x[indices, feature], kind='stable')
            sorted_indices = indices[order]
            column = x[sorted_indices, feature]
            positions = np.flatnonzero(column[:-1] < column[1:]) + 1
            positions = positions[(positions >= min_leaf) & (positions <= len(indices) - min_leaf)]
            if not len(positions):
                continue
            if len(positions) > candidates:
                positions = positions[np.unique(np.linspace(0, len(positions) - 1, candidates).astype(int))]
            sums = np.cumsum(y[sorted_indices], axis=0)[positions - 1]
            scores = (sums * sums).sum(axis=1) / positions + ((total - sums) ** 2).sum(axis=1) / (len(indices) - positions)
            slot = int(np.argmax(scores))
            gain = float(scores[slot] - parent)
            if gain > 1e-12 and (best is None or gain > best[0] + 1e-12):
                position = int(positions[slot])
                # Midpoints can round onto the right value for adjacent floats.
                # Using the observed left endpoint preserves the fitted partition exactly.
                threshold = float(column[position - 1])
                best = (gain, int(feature), threshold, sorted_indices[:position], sorted_indices[position:])
        if best is not None:
            gain, feature, threshold, left, right = best
            node.update(feature=feature, threshold=threshold, gain=gain,
                        left=grow(left, level + 1), right=grow(right, level + 1))
        return node

    return grow(np.arange(len(x)), 0)


def predict_tree(tree, features):
    x = np.atleast_2d(features)
    result = np.empty((len(x), len(tree['value'])))

    def visit(node, indices):
        if not len(indices):
            return
        if 'feature' not in node:
            result[indices] = node['value']
            return
        left = x[indices, node['feature']] <= node['threshold']
        visit(node['left'], indices[left])
        visit(node['right'], indices[~left])

    visit(tree, np.arange(len(x)))
    return result


def encode_values(values, scale):
    encoded = np.array(values, copy=True)
    encoded[:, 1:] = (values[:, 1:] - values[:, :1]) * scale
    return encoded


def decode_values(encoded, scale):
    values = np.array(encoded, copy=True)
    values[:, 1:] = values[:, :1] + encoded[:, 1:] / scale
    return values


def choose_actions(gross_q, features, config):
    """The known immediate fee enters exactly once, outside the fitted gross value."""
    q = np.atleast_2d(gross_q)
    x = np.atleast_2d(features)
    fees = action_fees(x, config)
    advantage = q - q[:, :1]
    benefit = -np.expm1(np.clip(-advantage, -700, 700))
    eligible = benefit > fees + config['learning_buffer']
    eligible[:, 0] = True
    net = q + np.log1p(-fees)
    net[~eligible] = -np.inf
    choices = np.argmax(net, axis=1)
    choices[(x[:, 11] > .5) | (x[:, 2] >= config['review_drawdown'])] = 0
    return choices, net[np.arange(len(x)), choices], benefit, fees


def counterfactual_data(scenarios, config):
    ids = [s.scenario_id for s in scenarios]
    if not ids or len(set(ids)) != len(ids) or any(not 1 <= sid <= 120 for sid in ids):
        raise PermissionError('FQI training requires unique development IDs')
    targets = np.array(config['learning_targets'])
    anchor = np.array((.5, .35, .15))
    output = {key: [] for key in ('x', 'next_x', 'gross_log', 'breach', 'fees',
                                  'scenario_id', 'period', 'behavior', 'available')}
    for scenario in scenarios:
        if not 2 <= len(scenario.observations) <= 30:
            raise ValueError('FQI training paths need 2 to 30 periods')
        for t, observation in enumerate(scenario.observations, 1):
            if observation.scenario_id != scenario.scenario_id or observation.period != t:
                raise ValueError('Invalid FQI observation sequence')
            if not all(np.isfinite(v) and v > -1 for v in (*observation.nominal_returns, observation.inflation)):
                raise ValueError('Invalid FQI market observation')
        for cost in (config['cost_rate'], 2 * config['cost_rate']):
            initial_fee = cost * (1 - anchor[2])
            first = scenario.observations[0]
            h = (1 - initial_fee) * anchor * (1 + np.array(first.nominal_returns)) / (1 + first.inflation)
            peak = max(1., float(h.sum()))
            maximum = max(initial_fee, 1 - h.sum() / peak)
            for behavior in range(4):
                holdings, high, mdd, pending = h.copy(), peak, maximum, False
                for t in range(2, len(scenario.observations) + 1):
                    observation = scenario.observations[t - 1]
                    value = holdings.sum()
                    p = holdings / value
                    drawdown = 1 - value / high
                    x = state_features(p, drawdown, mdd, scenario.observations[:t - 1], cost, pending)
                    forced = pending or drawdown >= config['review_drawdown']
                    next_states, logs, breaches, fees, next_holdings, next_peaks, next_mdds = [], [], [], [], [], [], []
                    real_factors = (1 + np.array(observation.nominal_returns)) / (1 + observation.inflation)
                    for action in range(4):
                        fee = 0. if action == 0 or forced else cost * .5 * np.abs(p - targets[action - 1]).sum()
                        post = holdings.copy() if action == 0 or forced else value * (1 - fee) * targets[action - 1]
                        end = post * real_factors
                        new_peak = max(high, float(post.sum()), float(end.sum()))
                        new_mdd = max(mdd, 1 - post.sum() / high, 1 - end.sum() / new_peak)
                        next_states.append(state_features(end / end.sum(), 1 - end.sum() / new_peak, new_mdd,
                            scenario.observations[:t], cost, forced))
                        logs.append(float(np.log(end.sum() / post.sum())))
                        breaches.append(float(mdd <= config['learning_loss_tolerance'] < new_mdd))
                        fees.append(float(fee))
                        next_holdings.append(end); next_peaks.append(new_peak); next_mdds.append(new_mdd)
                    for key, item in dict(x=x, next_x=next_states, gross_log=logs, breach=breaches, fees=fees,
                        scenario_id=scenario.scenario_id, period=t, behavior=behavior,
                        available=len(scenario.observations) - t + 1).items():
                        output[key].append(item)
                    holdings, high, mdd, pending = next_holdings[behavior], next_peaks[behavior], next_mdds[behavior], forced
    arrays = {key: np.asarray(values, dtype=np.int64 if key in ('scenario_id', 'period', 'behavior', 'available') else np.float64)
              for key, values in output.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError('Nonfinite counterfactual transition')
    return arrays


def fit_fqi(data, penalty, feature_set, config):
    started = time.perf_counter()
    features = list(range(len(FEATURE_NAMES))) if feature_set == 'full' else list(PORTFOLIO_FEATURES)
    models, trace = {}, []
    scale = config['learning_advantage_scale']
    for remaining in range(1, int(data['available'].max()) + 1):
        mask = data['available'] >= remaining
        x, nx = data['x'][mask], data['next_x'][mask]
        target = data['gross_log'][mask] - penalty * data['breach'][mask]
        if remaining > 1:
            previous = models[str(remaining - 1)]['teacher']
            future_x = nx.reshape(-1, len(FEATURE_NAMES))
            future_q = decode_values(predict_tree(previous, future_x), scale)
            _, continuation, _, _ = choose_actions(future_q, future_x, config)
            target = target + continuation.reshape(-1, 4)
        encoded = encode_values(target, scale)
        teacher = fit_tree(x, encoded, config['learning_teacher_depth'], config['learning_min_leaf'],
                           config['learning_split_candidates'], features)
        teaching = predict_tree(teacher, x)
        student = fit_tree(x, teaching, config['learning_student_depth'], config['learning_min_leaf'],
                           config['learning_split_candidates'], features)
        teacher_q, student_q = decode_values(teaching, scale), decode_values(predict_tree(student, x), scale)
        teacher_choices = choose_actions(teacher_q, x, config)[0]
        student_choices = choose_actions(student_q, x, config)[0]
        trace.append(dict(remaining=remaining, rows=len(x), target_sum=target.sum(axis=0).tolist(),
            target_square_sum=(target ** 2).sum(axis=0).tolist(),
            teacher_rmse=float(np.sqrt(np.mean((teacher_q - target) ** 2))),
            distillation_rmse=float(np.sqrt(np.mean((student_q - teacher_q) ** 2))),
            training_action_agreement=float(np.mean(teacher_choices == student_choices))))
        models[str(remaining)] = dict(teacher=teacher, student=student)
    return dict(feature_set=feature_set, feature_indices=features, penalty=penalty,
        train_ids=sorted(set(data['scenario_id'].tolist())), dataset_sha256=array_digest(data),
        source_states=len(data['x']), counterfactual_transitions=4 * len(data['x']),
        maximum_remaining=max(map(int, models)), models=models, trace=trace,
        fit_seconds=time.perf_counter() - started)


def fit_learning_models(scenarios, config):
    data = counterfactual_data(scenarios, config)
    fitted = {feature_set: {str(penalty): fit_fqi(data, penalty, feature_set, config)
        for penalty in config['learning_penalties']} for feature_set in ('full', 'portfolio')}
    return fitted


def save_training_data(scenarios, config, fitted, path):
    data = counterfactual_data(scenarios, config)
    if array_digest(data) != next(iter(fitted['full'].values()))['dataset_sha256']:
        raise ValueError('Counterfactual dataset does not match fitted model')
    with path.open('xb') as handle:
        np.savez_compressed(handle, **data)


class FQIPolicy:
    requires_benefit_gate = True

    def __init__(self, name, model, params, config, horizon):
        if name not in LEARNING_NAMES or horizon is None or not 2 <= horizon <= 30:
            raise ValueError('FQI requires an explicit evaluation horizon')
        self.name, self.horizon, self.config = name, horizon, config
        self.initial_target = (.5, .35, .15)
        self.buffer = config['learning_buffer']
        feature_set = 'portfolio' if name == 'fqi_portfolio' else 'full'
        self.fitted = model['fqi'][feature_set][str(params['penalty'])]
        if horizon - 1 > self.fitted['maximum_remaining']:
            raise ValueError('Evaluation horizon exceeds fitted value functions')
        self.kind = 'teacher' if name == 'fqi_teacher' else 'student'
        self.maximum_drawdown = 0.
        self.last_period = 1
        self.inference_seconds = 0.
        self.inference_calls = 0

    def decide(self, context):
        started = time.perf_counter()
        try:
            return self._decide(context)
        finally:
            self.inference_seconds += time.perf_counter() - started
            self.inference_calls += 1

    def _decide(self, context):
        if context.period != self.last_period + 1 or len(context.history) != context.period - 1:
            raise ValueError('FQI needs sequential causal contexts')
        self.last_period = context.period
        value = sum(context.holdings)
        drawdown = 1 - value / context.cumulative_inflation / context.real_peak
        self.maximum_drawdown = max(self.maximum_drawdown, drawdown,
                                    context.transaction_cost_rate * (1 - self.initial_target[2]))
        if drawdown >= self.config['review_drawdown']:
            return Decision(Action.HUMAN_REVIEW, ('DRAWDOWN_ALERT',), REVIEW_EXPLANATION)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid FQI wealth')
        weights = np.array(context.holdings) / value
        x = state_features(weights, drawdown, self.maximum_drawdown, context.history,
                           context.transaction_cost_rate, context.review_pending)
        if not np.isfinite(x).all():
            return Decision(Action.HUMAN_REVIEW, ('INVALID_ESTIMATE',),
                'The state estimate is unreliable. A human adviser should review your plan. We will hold your existing investments while review is pending.')
        remaining = self.horizon - context.period + 1
        tree = self.fitted['models'][str(remaining)][self.kind]
        q = decode_values(predict_tree(tree, x), self.config['learning_advantage_scale'])
        choice, _, benefit, fees = choose_actions(q, x, self.config)
        action = int(choice[0])
        if not action:
            return Decision(Action.KEEP, ('FQI_HOLD',),
                'The estimated long-term benefit does not justify a change after costs and the safety margin.')
        target = tuple(self.config['learning_targets'][action - 1])
        validate_weights(target)
        self.maximum_drawdown = max(self.maximum_drawdown, 1 - (1 - drawdown) * (1 - fees[0, action]))
        action_type = Action.REBALANCE if np.allclose(target, context.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        return Decision(action_type, ('FQI_VALUE_EXCEEDS_COST', 'TARGET_' + str(action)),
            'Your portfolio risk and recent market conditions favor this allocation in our training scenarios. The estimated long-term benefit exceeds the trading cost and safety margin. Losses remain possible.',
            target, float(benefit[0, action]), float(fees[0, action]))
