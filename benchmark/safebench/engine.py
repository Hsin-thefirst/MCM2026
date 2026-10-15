"""Auditable annual engine. Policies see immutable prefixes, never scenario objects."""
import math
import numpy as np
from safeinvest.contracts import (Action, Decision, PolicyContext, TradeResult, PeriodRecord,
                                  SimulationResult, validate_weights)

def execute_trade(holdings, target, cost_rate):
    validate_weights(target)
    h = np.asarray(holdings, dtype=float)
    if h.shape != (3,) or not np.isfinite(h).all() or (h < 0).any() or h.sum() <= 0:
        raise ValueError('Invalid holdings')
    if not math.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError('Invalid cost rate')
    v = h.sum()
    moved = .5 * np.abs(v*np.asarray(target)-h).sum()
    fee = cost_rate*moved
    return TradeResult(tuple((v-fee)*np.asarray(target)), float(moved), float(fee))

def simulate(scenario, policy, initial_target, config):
    validate_weights(initial_target)
    if len(scenario.observations) < config.horizon:
        raise ValueError('Insufficient observations')
    h = (0.,0.,config.initial_wealth)
    target = tuple(initial_target)
    inflation, peak, pending = 1., config.initial_wealth, False
    records = []
    initial = None
    for t in range(1, config.horizon+1):
        o = scenario.observations[t-1]
        if o.period != t or o.scenario_id != scenario.scenario_id:
            raise ValueError('Invalid observation sequence')
        if not all(math.isfinite(x) and x > -1 for x in (*o.nominal_returns,o.inflation)):
            raise ValueError('Invalid market observation')
        ctx = PolicyContext(t,scenario.observations[:t-1],h,target,inflation,peak,pending,
                            config.transaction_cost_rate)
        if t == 1:
            decision = Decision(Action.REBALANCE,('INITIAL_ALLOCATION',),'Initial allocation.',target)
        elif pending:
            decision = Decision(Action.KEEP,('REVIEW_PENDING',),'Hold existing assets while review is pending.')
        else:
            decision = policy.decide(ctx)
        if not isinstance(decision, Decision) or not isinstance(decision.action, Action):
            raise ValueError('Invalid policy decision')
        pre = h
        if decision.action in (Action.REBALANCE,Action.CHANGE_TARGET):
            if decision.action == Action.REBALANCE and not np.allclose(decision.target_weights,target,atol=1e-10,rtol=0):
                raise ValueError('Rebalance must restore the current target')
            trade = execute_trade(h,decision.target_weights,config.transaction_cost_rate)
            if t > 1 and getattr(policy,'requires_benefit_gate',False):
                actual_fraction = trade.transaction_cost/sum(h)
                if (decision.estimated_cost is None
                    or not math.isclose(decision.estimated_cost,actual_fraction,abs_tol=1e-10)):
                    raise ValueError('Trade cost estimate inconsistent with audited fee')
                if decision.gate_exempt:
                    # A benefit-gate exemption is one-sided: it is valid only for
                    # a strict reduction in equity exposure.  The check is kept in
                    # the accounting layer so policies cannot silently mislabel
                    # re-risking or neutral trades as protective.
                    pre_equity = float(pre[0] / sum(pre))
                    if not (float(decision.target_weights[0]) < pre_equity - 1e-12):
                        raise ValueError('Gate exemption requires strict equity reduction')
                elif (decision.estimated_benefit is None
                      or decision.estimated_benefit <= actual_fraction + getattr(policy,'buffer',0)):
                    raise ValueError('Trade fails auditable benefit/cost gate')
            target = tuple(decision.target_weights)
            h = trade.post_trade_holdings
        else:
            trade = TradeResult(h,0.,0.)
            if decision.action == Action.HUMAN_REVIEW:
                pending = True
        if initial is None:
            initial = trade
        # Include all fee-time marks in MDD/peak accounting, as well as annual marks.
        peak = max(peak,sum(h)/inflation)
        post = h
        h = tuple(x*(1+r) for x,r in zip(h,o.nominal_returns))
        inflation *= 1+o.inflation
        wealth = sum(h)/inflation
        if not math.isfinite(wealth) or wealth <= 0:
            raise ValueError('Invalid wealth path')
        peak = max(peak,wealth)
        records.append(PeriodRecord(t,decision.action,target,pre,post,h,trade.moved_amount,
                                    trade.transaction_cost,inflation,wealth,pending,
                                    decision.reason_codes,decision.explanation,
                                    decision.estimated_benefit,decision.estimated_cost,
                                    decision.gate_exempt))
    return SimulationResult(scenario.scenario_id,policy.name,config,tuple(initial_target),
                            initial.moved_amount,initial.transaction_cost,
                            initial.post_trade_holdings,tuple(records))
