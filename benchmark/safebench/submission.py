"""Bounded execution candidates and training-only selection for the submission."""
import numpy as np
import pandas as pd
from safeinvest.contracts import Action, Decision
from .plan_a import PlanAPolicy, selection_constraints
from .plan_b import PlanBPolicy
from .plan_c import PlanCPolicy

BASE_RULES = ['plan_a_balanced', 'submission_a_continuous', 'submission_a_band',
              'submission_c_full', 'submission_c_cap10', 'submission_c_cap20',
              'submission_b_long025', 'submission_b_long05']
POLICIES = BASE_RULES[1:] + ['submission_selected']
COMPLEXITY = [BASE_RULES[i] for i in (0, 1, 2, 6, 7, 4, 5, 3)]


def enabled(config):
    return config.get('submission', {}).get('enabled', False)


def continuous_fraction(c, q, f):
    """Maximize (1-f*l)*(c-q*(1-l)**2) on [0,1], before the buffer gate."""
    if q <= 0:
        return 0.
    if f == 0:
        return 1.
    d = 2*q*(1-f)
    x = 2*f*c/(d+np.sqrt(d*d+12*f*f*q*c))
    return float(np.clip(1-x, 0., 1.))


class ExecutionPolicy(PlanAPolicy):
    def __init__(self, name, model, params, config, horizon):
        super().__init__('plan_a_balanced', model, params, config, horizon)
        self.name = name
        self.last_diagnostics = None

    def _decide(self, c):
        self.last_diagnostics = None
        advice = super()._decide(c)
        if advice.action == Action.HUMAN_REVIEW:
            return advice
        current = np.asarray(c.holdings)/sum(c.holdings)
        gap = current-self.anchor
        constant = 1+self.anchor_mean
        q = max(0., float(.5*self.gamma*(gap@self.cov@gap)))
        full_cost = float(c.transaction_cost_rate*.5*np.abs(gap).sum())
        if self.execution_rule == 'submission_a_continuous':
            fractions = np.append(self.settings['move_fractions'], continuous_fraction(constant, q, full_cost))
        else:
            deviation = float(np.max(np.abs(gap)))
            band = self.settings['band']
            fractions = np.array([0., 1-band/deviation]) if deviation > band else np.array([0.])
        targets = current+fractions[:, None]*(self.anchor-current)
        values = constant-q*(1-fractions)**2
        costs = full_cost*fractions
        benefits = (values-values[0])/values
        allowed = benefits > costs+self.buffer
        # Holding is always available; old grid actions remain available in continuous mode.
        net = np.where(allowed, (1-costs)*values, -np.inf)
        net[0] = values[0]
        chosen = int(np.argmax(net))
        self.last_diagnostics = dict(execution_rule=self.execution_rule,
            chosen_fraction=float(fractions[chosen]), full_move_cost=full_cost,
            candidate_count=len(fractions), chosen_net_value=float(net[chosen]))
        if chosen == 0:
            return Decision(Action.KEEP, ('SUBMISSION_NO_ELIGIBLE_MOVE',),
                'The estimated improvement does not cover trading costs and the safety margin. Hold the existing investments.',
                estimated_benefit=0., estimated_cost=0.)
        return self._trade(c, targets[chosen], float(benefits[chosen]), float(costs[chosen]),
                           'SUBMISSION_COST_GATED_MOVE')


def make_submission_policy(name, model, params, config, horizon):
    rule = params.get('execution_rule') if name == 'submission_selected' else name
    if rule not in BASE_RULES:
        raise ValueError('Missing or unknown submission execution rule')
    if rule == 'plan_a_balanced':
        p = PlanAPolicy(rule, model, params, config, horizon)
    elif rule.startswith('submission_a_'):
        p = ExecutionPolicy(rule, model, params, config, horizon)
    elif rule.startswith('submission_c_'):
        p = PlanCPolicy('plan_c_l1', model, params, config, horizon)
        p.move_cap = {'submission_c_full': None, 'submission_c_cap10': .1,
                      'submission_c_cap20': .2}[rule]
    else:
        label = '0' if horizon <= 6 else ('025' if rule.endswith('025') else '05')
        p = PlanBPolicy('plan_b_narrow_'+label, model, params, config, horizon)
    p.name = name
    p.execution_rule = rule
    return p


def strategic_bill_min(policy):
    targets = [np.asarray(policy.initial_target)]
    if hasattr(policy, 'defensive'):
        targets.append(policy.defensive)
    return float(min(w[2] for w in targets))


def choose_rule(scores, config):
    rows = []
    eligible = []
    for index, name in enumerate(BASE_RULES):
        block = scores[scores.policy == name]
        violation, details = selection_constraints(block, config)
        score = float(np.log1p(block.real_cagr).mean())
        row = dict(policy='submission_selected', trial=index, execution_rule=name,
                   mean_utility=score, selection_feasible=violation <= 1e-10,
                   constraint_violation=violation, horizon_constraints=details)
        rows.append(row)
        if row['selection_feasible']:
            eligible.append(row)
    if not eligible:
        # This is an explicit service fallback, never a successful constraint screen.
        return dict(execution_rule='plan_a_balanced', execution_selection_feasible=False), rows
    best = max(eligible, key=lambda r: r['mean_utility'])
    pivot = scores.assign(log_growth=np.log1p(scores.real_cagr)).pivot(
        index=['scenario_id', 'horizon'], columns='policy', values='log_growth')
    near = []
    for row in eligible:
        diff = (pivot[best['execution_rule']]-pivot[row['execution_rule']]).groupby('scenario_id').mean()
        se = float(diff.std(ddof=1)/np.sqrt(len(diff)))
        row.update(paired_gap_se=se, one_se_eligible=bool(diff.mean() <= se+1e-12))
        if row['one_se_eligible']:
            near.append(row)
    winner = min(near, key=lambda r: COMPLEXITY.index(r['execution_rule']))
    return dict(execution_rule=winner['execution_rule'], execution_selection_feasible=True), rows


def tune_submission(inner_models, anchor_params, config, evaluate):
    blocks = []
    evidence = []
    names = BASE_RULES+['plan_a_annual']
    for model, valid in inner_models:
        metrics, _ = evaluate(valid, model, {n: anchor_params for n in names}, config,
                              names=names, keep_records=False)
        reference = metrics[metrics.policy == 'plan_a_annual'][
            ['scenario_id', 'horizon', 'real_cagr']].rename(columns={'real_cagr': 'annual_growth'})
        block = metrics[metrics.policy.isin(BASE_RULES)].merge(
            reference, on=['scenario_id', 'horizon'], validate='many_to_one')
        columns = ['policy', 'scenario_id', 'horizon', 'real_cagr', 'real_max_drawdown',
                   'review', 'plan_a_risk_budget', 'plan_a_validation_budget', 'annual_growth']
        block = block[columns]
        blocks.append(block)
        evidence.append(block.to_dict('records'))
    selected, rows = choose_rule(pd.concat(blocks, ignore_index=True), config)
    return dict(anchor_params, **selected), rows, evidence
