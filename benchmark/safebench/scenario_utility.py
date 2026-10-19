"""Training-only block valuation and fixed-target, auditable CRRA cost gates."""
import numpy as np
from scipy.special import logsumexp
from safeinvest.contracts import Action, Decision, validate_weights

CE_POLICY_NAMES = ('utility_band_1y', 'utility_band_3y')
CE_HORIZONS = dict(zip(CE_POLICY_NAMES, (1, 3)))
REVIEW_EXPLANATION = ('Your purchasing-power loss has crossed our review threshold. '
    'A human adviser should reassess your plan. We will hold your existing investments while review is pending.')


def fit_utility_blocks(scenarios, horizons=(1, 3)):
    ids = [s.scenario_id for s in scenarios]
    if not ids or len(set(ids)) != len(ids) or any(not 1 <= sid <= 120 for sid in ids):
        raise PermissionError('Utility fitting requires unique development scenario IDs')
    blocks = {}
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Invalid valuation horizon')
        gross, probabilities, source_ids, starts = [], [], [], []
        for scenario in scenarios:
            observations = scenario.observations
            count = len(observations) - horizon + 1
            if count <= 0:
                raise ValueError('Training path is shorter than valuation horizon')
            for year, observation in enumerate(observations, 1):
                values = (*observation.nominal_returns, observation.inflation)
                if observation.scenario_id != scenario.scenario_id or observation.period != year:
                    raise ValueError('Invalid training observation sequence')
                if not all(np.isfinite(x) and x > -1 for x in values):
                    raise ValueError('Invalid training return/inflation')
            real_gross = np.array([[(1 + r) / (1 + o.inflation) for r in o.nominal_returns]
                                   for o in observations])
            for start in range(count):
                factor = np.prod(real_gross[start:start + horizon], axis=0)
                if not np.isfinite(factor).all() or (factor <= 0).any():
                    raise ValueError('Nonfinite training block')
                gross.append(factor.tolist())
                probabilities.append(1 / (len(scenarios) * count))
                source_ids.append(scenario.scenario_id)
                starts.append(start + 1)
        blocks[str(horizon)] = dict(horizon=horizon, gross=gross, probabilities=probabilities,
            scenario_ids=source_ids, start_periods=starts)
    return blocks


def log_certainty_equivalent(gross, weights, gamma, probabilities):
    gross = np.asarray(gross, dtype=float)
    validate_weights(tuple(weights))
    if gross.ndim != 2 or gross.shape[1] != 3 or not np.isfinite(gross).all() or (gross <= 0).any():
        raise ValueError('Invalid scenario growth factors')
    values = gross @ np.asarray(weights, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (not len(values) or not np.isfinite(values).all() or (values <= 0).any()
            or probabilities.shape != values.shape or not np.isfinite(probabilities).all()
            or (probabilities <= 0).any() or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10)
            or not np.isfinite(gamma) or gamma < 0):
        raise ValueError('Invalid certainty-equivalent estimate')
    logs = np.log(values)
    if gamma == 1:
        return float(probabilities @ logs)
    return float(logsumexp((1 - gamma) * logs, b=probabilities) / (1 - gamma))


class ScenarioUtilityPolicy:
    requires_benefit_gate = True

    def __init__(self, name, blocks, params, config, anchor):
        self.name = name
        self.initial_target = tuple(anchor)
        validate_weights(self.initial_target)
        self.band, self.buffer = params['band'], params['buffer']
        self.gamma = config['ce_gamma']
        self.review_threshold = config['review_drawdown']
        if (not np.isfinite(self.band) or not 0 <= self.band < 1
                or not np.isfinite(self.buffer) or self.buffer < 0):
            raise ValueError('Invalid band/buffer')
        self.gross = np.asarray(blocks['gross'], dtype=float)
        self.probabilities = np.asarray(blocks['probabilities'], dtype=float)
        self.target_log_ce = log_certainty_equivalent(self.gross, anchor, self.gamma, self.probabilities)

    def decide(self, context):
        if 1 - sum(context.holdings) / context.cumulative_inflation / context.real_peak >= self.review_threshold:
            return Decision(Action.HUMAN_REVIEW, ('DRAWDOWN_ALERT',), REVIEW_EXPLANATION)
        current = np.asarray(context.holdings) / sum(context.holdings)
        if np.max(np.abs(current - self.initial_target)) <= self.band:
            return Decision(Action.KEEP, ('WITHIN_TRADE_BAND',), 'The portfolio remains inside its agreed trading band.')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                current_log_ce = log_certainty_equivalent(self.gross, current, self.gamma, self.probabilities)
                benefit = float(-np.expm1(current_log_ce - self.target_log_ce))
            if not np.isfinite(benefit):
                raise ValueError('Nonfinite utility benefit')
        except (ValueError, FloatingPointError):
            return Decision(Action.HUMAN_REVIEW, ('INVALID_ESTIMATE',),
                'The scenario valuation is unreliable. A human adviser should review your plan. We will hold your existing investments while review is pending.')
        cost = float(context.transaction_cost_rate * .5 * np.abs(current - self.initial_target).sum())
        if benefit > cost + self.buffer:
            return Decision(Action.REBALANCE, ('SCENARIO_VALUE_EXCEEDS_COST',),
                'Your portfolio has drifted from its agreed mix. Our scenario estimates suggest restoring it is worth more than the trading cost and safety margin. Losses remain possible.',
                self.initial_target, benefit, cost)
        return Decision(Action.KEEP, ('BENEFIT_BELOW_COST',),
            'The scenario estimate does not justify the trading cost and safety margin.',
            estimated_benefit=benefit, estimated_cost=cost)


class ReviewAnnualPolicy:
    """Unconditional diagnostic with the same absorbing human-review rule."""
    name = 'balanced_annual_review'
    requires_benefit_gate = False

    def __init__(self, anchor, threshold):
        self.initial_target = tuple(anchor)
        self.threshold = threshold

    def decide(self, context):
        if 1 - sum(context.holdings) / context.cumulative_inflation / context.real_peak >= self.threshold:
            return Decision(Action.HUMAN_REVIEW, ('DRAWDOWN_ALERT',), REVIEW_EXPLANATION)
        return Decision(Action.REBALANCE, ('COMPARATOR_REBALANCE',),
            'Restore the fixed target allocation.', self.initial_target)
