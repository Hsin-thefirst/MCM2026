"""Execution-efficiency ablations; preserve frozen A and all prior research code."""
from dataclasses import replace
import numpy as np
from safeinvest.contracts import Action, Decision
from safebench.plan_a import PlanAPolicy
from fusion_policy import FusionPolicy, hold_values


class EfficientFusion(FusionPolicy):
    def __init__(self, name, model, params, config, horizon, settings, rule):
        super().__init__('fusion_drawdown', model, params, config, horizon, settings)
        self.name = name
        self.adviser = PlanAPolicy('plan_a_balanced', model, params, config, horizon)
        self.rule = rule
        self.buffer = rule['buffer']
        if 'annual_buffer' in rule:
            self.buffer = float(-np.expm1(-rule['annual_buffer']*horizon))
        self.use_a = horizon <= rule.get('short_a_max_horizon',0)
        if self.use_a: self.buffer = self.adviser.buffer
        if not .0001 <= self.buffer < 1: raise ValueError('Invalid execution buffer')
        self.last_trade_period = -100
        self.last_equity_direction = 0

    def _decide(self, c):
        self.last_diagnostics = None
        advice = self.adviser._decide(c)
        if self.use_a: return advice
        if advice.action == Action.HUMAN_REVIEW: return advice
        current = np.asarray(c.holdings)/sum(c.holdings)
        targets = self.candidates(c, advice)
        dd = max(0., 1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
        future = self.future[:, c.period-1:self.horizon]
        scale = len(future[0]) if self.rule.get('remaining_scale',False) else self.horizon
        q, fees, risk = hold_values(targets, current, dd, c.historical_max_drawdown,
                                   future, scale, c.transaction_cost_rate, self.fusion)
        if not np.isfinite(q).all():
            return Decision(Action.HUMAN_REVIEW, ('EFFICIENT_INVALID_VALUE',),
                'The allocation estimate is unreliable. Existing investments remain in place while a human adviser reviews your plan.')
        benefit = fees+(1-fees)*(-np.expm1(np.clip(-scale*(q-q[0]),-100,100)))
        allowed = benefit > fees+self.buffer
        direction = np.sign(targets[:,0]-current[0])
        reversal = (direction*self.last_equity_direction < 0) & (c.period-self.last_trade_period <= self.rule.get('reverse_wait',0))
        allowed &= ~reversal
        allowed[0] = True
        chosen = int(np.argmax(np.where(allowed,q,-np.inf)))
        self.last_diagnostics = dict(values=q.tolist(), costs=fees.tolist(), benefits=benefit.tolist(),
            targets=targets.tolist(), chosen=chosen, required_buffer=self.buffer, move_cap=self.fusion['move_cap'],
            reversal_blocked=reversal.tolist(), previous_trade_period=self.last_trade_period,
            previous_equity_direction=self.last_equity_direction,
            historical_max_drawdown=c.historical_max_drawdown)
        if chosen == 0:
            return Decision(Action.KEEP, ('EFFICIENT_HOLD',),
                'Keep existing investments: projected improvement is too small or reverses a recent adjustment.',
                estimated_benefit=0.,estimated_cost=0.)
        self.last_trade_period=c.period
        self.last_equity_direction=int(direction[chosen])
        decision=self._trade(c,targets[chosen],float(benefit[chosen]),float(fees[chosen]),'EFFICIENT_NET_VALUE')
        return replace(decision,explanation='This adjustment improves projected purchasing-power growth and drawdown risk after trading costs and a safety margin. Projections are uncertain; investment losses remain possible.')
