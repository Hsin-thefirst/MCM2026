"""Research-only adapters of Cai actions and frontier MPC on frozen OOF A anchors.

Sources: Cai f16a8aa policies.DrawdownBudgetPolicy and MPC 967c8a9
frontier.value_estimates. No gate exemptions or customer-rule relaxation.
"""
from dataclasses import replace
import numpy as np
from safeinvest.contracts import Action, Decision
from safebench.plan_a import PlanAPolicy


def tail(values, alpha=.9):
    ordered = np.sort(values, axis=-1)[..., ::-1]
    mass = ordered.shape[-1]*(1-alpha)
    k = int(np.floor(mass))
    return (ordered[..., :k].sum(axis=-1)+(mass-k)*ordered[..., k])/mass


def hold_values(targets, current, drawdown, maximum, future, horizon, cost, settings):
    """Buy once, hold in training futures; replan at the next real decision.

    All candidates share the continuation. Match actual calendar positions;
    history, fee-time losses and historical maximum loss enter every path.
    """
    fees = cost*.5*np.abs(targets-current).sum(axis=1)
    gross = np.cumprod(1+future, axis=1)
    wealth = np.einsum('ak,stk->ast', targets, gross)*(1-fees[:, None, None])
    marks = np.concatenate((np.broadcast_to(1-fees[:, None, None],
                            (len(targets), len(future), 1)), wealth), axis=2)
    peaks = np.maximum(np.maximum.accumulate(marks, axis=2), 1/(1-drawdown))
    losses = np.maximum((1-marks/peaks).max(axis=2), maximum)
    p90 = np.quantile(losses, .9, axis=1)
    risk = tail(losses)
    q = np.log(wealth[:, :, -1]).mean(axis=1)/horizon
    q -= settings['drawdown_penalty']*(risk-maximum)
    # Index 1 is the same capped strategic action for every ablation.
    q -= settings['risk_barrier']*np.maximum(0., p90-p90[1]-settings['risk_slack'])
    return q, fees, risk


class FusionPolicy(PlanAPolicy):
    def __init__(self, name, model, params, config, horizon, settings):
        if name not in ('fusion_mpc', 'fusion_drawdown', 'fusion_lifecycle'):
            raise ValueError('Unknown fusion policy')
        super().__init__('plan_a_balanced', model, params, config, horizon)
        self.name, self.horizon, self.fusion = name, horizon, settings
        control = model['scenario_control']
        if control['train_ids'] != model['train_ids'] or any(i not in range(1, 121) for i in model['train_ids']):
            raise ValueError('Fusion requires matching development training IDs')
        self.future = np.asarray(control['real_returns'])
        if self.future.shape != (len(model['train_ids']), 30, 3) or not np.isfinite(self.future).all() or (self.future <= -1).any():
            raise ValueError('Invalid fusion training paths')
        self.defensive = np.asarray(control['defensive'][str(horizon)])
        self.last_diagnostics = None

    def candidates(self, c, advice):
        current = np.asarray(c.holdings)/sum(c.holdings)
        advised = current if advice.target_weights is None else np.asarray(advice.target_weights)
        targets = [current, self.anchor, advised, self.defensive]
        if self.name != 'fusion_mpc':
            dd = max(0., 1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
            floor = min(self.fusion['equity_floor'], self.anchor[0])
            cap = self.anchor[0]
            if self.name == 'fusion_lifecycle':
                cap = floor+(cap-floor)*min(1., self.horizon/self.fusion['lifecycle_scale'])
            equity = floor+(cap-floor)*max(0., 1-dd/self.fusion['loss_budget'])**self.fusion['curvature']
            # Preserve A's defensive split; Cai's 70/30 split would change the anchor.
            split = self.anchor[1:]/self.anchor[1:].sum() if self.anchor[1:].sum() else np.array([0., 1.])
            targets.append(np.r_[equity, (1-equity)*split])
        targets = np.asarray(targets)
        moved = .5*np.abs(targets-current).sum(axis=1)
        scale = np.minimum(1., self.fusion['move_cap']/np.maximum(moved, 1e-300))
        return current+scale[:, None]*(targets-current)

    def _decide(self, c):
        self.last_diagnostics = None
        advice = super()._decide(c)
        if advice.action == Action.HUMAN_REVIEW:
            return advice
        current = np.asarray(c.holdings)/sum(c.holdings)
        targets = self.candidates(c, advice)
        dd = max(0., 1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
        q, fees, risk = hold_values(targets, current, dd, c.historical_max_drawdown,
            self.future[:, c.period-1:self.horizon], self.horizon, c.transaction_cost_rate, self.fusion)
        if not np.isfinite(q).all():
            return Decision(Action.HUMAN_REVIEW, ('FUSION_INVALID_VALUE',),
                'The allocation estimate is unreliable. Existing investments remain in place while a human adviser reviews your plan.')
        benefit = fees+(1-fees)*(-np.expm1(np.clip(-self.horizon*(q-q[0]), -100, 100)))
        allowed = benefit > fees+self.buffer
        allowed[0] = True
        winner = int(np.argmax(np.where(allowed, q, -np.inf)))
        self.last_diagnostics = dict(values=q.tolist(), costs=fees.tolist(), benefits=benefit.tolist(),
            targets=targets.tolist(), tail_drawdowns=risk.tolist(), chosen=winner,
            historical_max_drawdown=c.historical_max_drawdown, move_cap=self.fusion['move_cap'])
        if winner == 0:
            return Decision(Action.KEEP, ('FUSION_HOLD',),
                'Keep existing investments: no projected risk-adjusted improvement covers trading costs and the safety margin.',
                estimated_benefit=0., estimated_cost=0.)
        decision = self._trade(c, targets[winner], float(benefit[winner]), float(fees[winner]), 'FUSION_PATH_VALUE')
        return replace(decision, explanation='This adjustment improves projected purchasing-power growth and drawdown risk after trading costs. The projection uses past training scenarios and is uncertain. Investment losses remain possible.')
