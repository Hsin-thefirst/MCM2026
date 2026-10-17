"""Shared-action finite scenario MPC with a frozen Plan A continuation."""
from dataclasses import replace
from itertools import product
import numpy as np
from safeinvest.contracts import Action, Decision
from .plan_a import PlanAPolicy, log_path_values

POLICIES = ['plan_c_l1', 'plan_c_l3', 'plan_c_l1_no_risk', 'plan_c_l3_no_risk']


def enabled(config): return config.get('plan_c', {}).get('enabled', False)


def fit_control(scenarios, config):
    from .policies import grid
    if not scenarios or any(s.scenario_id > 120 for s in scenarios):
        raise PermissionError('Scenario control accepts development paths only')
    returns = np.asarray([[(1+np.asarray(o.nominal_returns))/(1+o.inflation)-1 for o in s.observations] for s in scenarios])
    weights = grid(bounds=((0., 1.),)*3)
    defensive = {}
    for h in config['horizons']:
        risks = [log_path_values(returns[:, :h], w, config['plan_a']['alpha'])['risk'] for w in weights]
        defensive[str(h)] = weights[int(np.argmin(risks))].tolist()
    return dict(train_ids=[s.scenario_id for s in scenarios], real_returns=returns.tolist(), defensive=defensive)


def batch_a(current, policy, rate):
    fractions = np.asarray(policy.settings['move_fractions'])
    gap = current-policy.anchor
    quadratic = np.einsum('...i,ij,...j->...', gap, policy.cov, gap)
    values = 1+policy.anchor_mean-.5*policy.gamma*quadratic[..., None]*(1-fractions)**2
    costs = rate*.5*np.abs(gap).sum(axis=-1)[..., None]*fractions
    invalid = (~np.isfinite(values)).any(axis=-1) | (values<=0).any(axis=-1)
    choice = np.argmax((1-costs)*values, axis=-1)
    chosen_value = np.take_along_axis(values, choice[..., None], axis=-1)[..., 0]
    cost = np.take_along_axis(costs, choice[..., None], axis=-1)[..., 0]
    benefit = (chosen_value-values[..., 0])/chosen_value
    fraction = np.where(benefit > cost+policy.buffer, fractions[choice], 0.)
    return current+fraction[..., None]*(policy.anchor-current), invalid


def rollout(policy, c, depth, kappa):
    future = policy.future[:, c.period-1:policy.horizon]
    depth = min(depth, future.shape[1])
    sequences = np.asarray(list(product(range(4), repeat=depth)), dtype=int)
    count, n = len(sequences), len(future)
    current = np.asarray(c.holdings)/sum(c.holdings)
    wealth_now = sum(c.holdings)/c.cumulative_inflation
    holdings = np.broadcast_to(current*wealth_now, (count, n, 3)).copy()
    peak = np.full((count, n), c.real_peak)
    mdd = np.full((count, n), c.historical_max_drawdown)
    pending = np.full((count, n), c.review_pending)
    total_cost = np.zeros((count, n))
    first_targets = None; first_costs = None
    for step in range(future.shape[1]):
        wealth = holdings.sum(axis=-1)
        current = holdings/wealth[..., None]
        pending |= 1-wealth/peak >= policy.review_threshold
        advice, invalid = batch_a(current, policy, c.transaction_cost_rate)
        pending |= invalid
        if step < depth:
            kind = sequences[:, step, None, None]
            target = np.where(kind == 0, current, np.where(kind == 1, advice,
                      np.where(kind == 2, policy.anchor, policy.defensive)))
        else:
            target = advice
        target = np.where(pending[..., None], current, target)
        fraction = c.transaction_cost_rate*.5*np.abs(target-current).sum(axis=-1)
        if step == 0:
            first_targets = target[:, 0].copy(); first_costs = fraction[:, 0].copy()
        total_cost += fraction
        post_wealth = wealth*(1-fraction)
        mdd = np.maximum(mdd, 1-post_wealth/peak)
        peak = np.maximum(peak, post_wealth)
        holdings = post_wealth[..., None]*target*(1+future[None, :, step, :])
        wealth = holdings.sum(axis=-1)
        mdd = np.maximum(mdd, 1-wealth/peak)
        peak = np.maximum(peak, wealth)
    growth = np.log(wealth/wealth_now).mean(axis=1)
    losses = -np.log1p(-mdd)
    mass = n*(1-policy.alpha); whole = int(np.floor(mass))
    order = np.sort(losses, axis=1)[:, ::-1]
    risk = (order[:, :whole].sum(axis=1)+(mass-whole)*order[:, whole])/mass if whole < n else order.mean(axis=1)
    objective = growth-kappa*risk
    return dict(sequences=sequences, objective=objective, growth=growth, risk=risk,
                targets=first_targets, costs=first_costs, future_cost=total_cost.mean(axis=1))


class PlanCPolicy(PlanAPolicy):
    def __init__(self, name, model, params, config, horizon):
        super().__init__('plan_a_balanced', model, params, config, horizon)
        self.name = name; self.horizon = horizon
        self.depth = 3 if '_l3' in name else 1
        self.kappa = 0. if name.endswith('_no_risk') else 1.
        fitted = model['scenario_control']
        if fitted['train_ids'] != model['train_ids']: raise ValueError('Control training IDs differ')
        self.future = np.asarray(fitted['real_returns'])
        self.defensive = np.asarray(fitted['defensive'][str(horizon)])
        self.alpha = config['plan_a']['alpha']; self.last_diagnostics = None

    def _decide(self, c):
        self.last_diagnostics = None
        advice = super()._decide(c)
        if advice.action == Action.HUMAN_REVIEW:
            return advice
        result = rollout(self, c, self.depth, self.kappa)
        j = result['objective']; seq = result['sequences']
        winner = int(np.argmax(j)); hold = float(np.max(j[seq[:, 0] == 0]))
        cost = float(result['costs'][winner])
        # The objective already includes all fees. This split recovers the gate's units.
        benefit = float(cost+(1-cost)*(-np.expm1(hold-j[winner])))
        self.last_diagnostics = dict(depth=self.depth, kappa=self.kappa, scenarios=len(self.future),
            sequence_count=len(seq), chosen_sequence=seq[winner].tolist(), best_objective=float(j[winner]),
            hold_objective=hold, mean_log_growth=float(result['growth'][winner]),
            log_drawdown_es=float(result['risk'][winner]), first_cost=cost, benefit=benefit,
            mean_future_cost_fraction_sum=float(result['future_cost'][winner]),
            historic_max_drawdown=c.historical_max_drawdown,
            first_action_values=[float(np.max(j[seq[:, 0] == a])) for a in range(4)])
        if seq[winner, 0] == 0 or benefit <= cost+self.buffer:
            return Decision(Action.KEEP, ('PLAN_C_BELOW_COST',), 'The projected net improvement does not cover the safety margin. Hold the existing investments.', estimated_benefit=benefit, estimated_cost=cost)
        decision = self._trade(c, result['targets'][winner], benefit, cost, 'PLAN_C_NET_PATH_VALUE')
        return replace(decision, explanation='This adjustment has higher projected value after accounting for trading costs and potential purchasing-power losses. Projections are uncertain; investment losses remain possible.')
