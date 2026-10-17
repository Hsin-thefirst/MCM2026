"""Fixed ridge tilt with matched action-family controls and training-only state."""
import numpy as np
from safeinvest.contracts import Action, Decision
from .plan_a import PlanAPolicy
from .signals import forecast

FIXED = {f'plan_b_{family}_{label}': (family, eta, 'ridge')
         for family in ('narrow', 'wide') for label, eta in (('0', 0.), ('025', .25), ('05', .5))}
FIXED.update(plan_b_period=('narrow', .5, 'period'), plan_b_shifted=('narrow', .5, 'shifted'))
POLICIES = list(FIXED) + ['plan_b_narrow', 'plan_b_wide']


def enabled(config): return config.get('plan_b', {}).get('enabled', False)


class PlanBPolicy(PlanAPolicy):
    def __init__(self, name, model, params, config, horizon):
        super().__init__('plan_a_balanced', model, params, config, horizon)
        self.name = name
        self.family, self.eta, self.signal = FIXED.get(name, (name.removeprefix('plan_b_'), params.get('eta', 0.), 'ridge'))
        self.fitted = model['signals']
        if self.fitted['train_ids'] != model['train_ids']: raise ValueError('Signal training IDs differ')
        self.ood_limit = config['plan_b']['ood_limit']
        self.last_diagnostics = None

    def _decide(self, c):
        delta, fallback, prediction = forecast(c.history, self.fitted, self.signal, self.ood_limit)
        self.last_diagnostics = dict(signal_model=self.signal, eta=self.eta, delta=delta,
            prediction=prediction, fallback=fallback, warmup=len(c.history)<2, family=self.family)
        # Exact original operations and decisions for the matched zero-tilt control.
        if (self.family == 'narrow' and self.eta == 0) or len(c.history) < 2 or fallback:
            return super()._decide(c)
        current = np.asarray(c.holdings)/sum(c.holdings)
        if 1-sum(c.holdings)/c.cumulative_inflation/c.real_peak >= self.review_threshold:
            return super()._decide(c)
        targets = list(current+np.asarray(self.settings['move_fractions'])[:, None]*(self.anchor-current))
        if self.family == 'wide':
            for i in range(3):
                for j in range(i+1, 3):
                    direction = np.eye(3)[i]-np.eye(3)[j]
                    for sign in (-1, 1):
                        candidate = self.anchor+sign*.05*direction
                        if (candidate >= -1e-12).all() and (candidate <= 1+1e-12).all():
                            targets.append(np.maximum(candidate, 0.))
        targets = np.asarray(targets)
        gap = targets-self.anchor
        values = 1+self.anchor_mean-.5*self.gamma*np.einsum('ij,jk,ik->i', gap, self.cov, gap)
        values += self.eta*delta*gap[:, 0]
        costs = c.transaction_cost_rate*.5*np.abs(targets-current).sum(axis=1)
        if not np.isfinite(values).all() or (values <= 0).any():
            return Decision(Action.HUMAN_REVIEW, ('PLAN_B_INVALID_VALUE',), 'The return estimate requires human review. Existing assets will be held.')
        index = int(np.argmax((1-costs)*values))
        benefit = float((values[index]-values[0])/values[index]); cost = float(costs[index])
        self.last_diagnostics.update(candidate_count=len(targets), chosen_index=index)
        if benefit <= cost+self.buffer:
            return Decision(Action.KEEP, ('PLAN_B_BELOW_COST',), 'The estimated improvement does not cover trading costs and the safety margin.', estimated_benefit=benefit, estimated_cost=cost)
        return self._trade(c, targets[index], benefit, cost, 'PLAN_B_SIGNAL_TILT')
