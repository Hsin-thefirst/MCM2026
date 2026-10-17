"""Generated-state checks for bounded execution, selection and suitability."""
from dataclasses import replace
import copy
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from safeinvest.contracts import Action, PolicyContext, SimulationConfig
from safebench.engine import execute_trade, simulate
from safebench.experiment import evaluation_schedule, suitability
from safebench.metrics import expected_shortfall
from safebench.plan_a import PlanAPolicy
from safebench.plan_c import rollout
from safebench.submission import (BASE_RULES, POLICIES, continuous_fraction,
                                  make_submission_policy, choose_rule, tune_submission)
from test_plan_bc import fixture


class SubmissionTests(unittest.TestCase):
    def test_promotion_screen_retains_baseline_without_evidence(self):
        from safebench.submission_analysis import comparisons_and_screen
        _, config, _, _ = fixture()
        config['bootstrap'] = 50
        names = BASE_RULES+['submission_selected', 'balanced_annual', 'plan_a_annual', 'equal_annual']
        metrics = pd.DataFrame([dict(policy=name, scenario_id=s, fold=s % 5, horizon=30,
            cost_rate=cost, real_cagr=.03+s/10000, real_max_drawdown=.2+s/1000, review=False)
            for name, s, cost in product(names, range(1, 13), (.0025, .005))])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics.to_csv(out/'scenario_metrics.csv', index=False)
            (out/'fitted_models.json').write_text(json.dumps([dict(params={
                name:dict(execution_selection_feasible=True) for name in POLICIES})]))
            pd.DataFrame(dict(risk_order_ok=[True])).to_csv(out/'suitability_risk.csv', index=False)
            comparisons_and_screen(out, config)
            result = json.loads((out/'submission_decision.json').read_text())
            self.assertEqual(result['selected_policy'], 'plan_a_balanced')
            self.assertFalse(result['adopted_new_execution'])
            screen = pd.read_csv(out/'submission_screen.csv')
            self.assertFalse(screen.growth_route.any())
            self.assertFalse(screen.risk_route.any())

    def test_inner_pipeline_retains_every_validation_case(self):
        from safebench.experiment import evaluate
        scenarios, config, model, params = fixture()
        config['horizons'] = [5, 30]
        valid = [replace(s, scenario_id=100+i, observations=tuple(
            replace(o, scenario_id=100+i) for o in s.observations)) for i, s in enumerate(scenarios[:3])]
        selected, rows, evidence = tune_submission([(model, valid)], params, config, evaluate)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(evidence[0]), 8*3*2)
        self.assertEqual(set(r['scenario_id'] for r in evidence[0]), {100, 101, 102})
        p = make_submission_policy('submission_selected', model, selected, config, 5)
        simulate(valid[0], p, p.initial_target, SimulationConfig(5))

    def test_continuous_global_objective_and_buffer_dominance(self):
        rng = np.random.default_rng(20261017)
        _, config, model, params = fixture()
        p = make_submission_policy('submission_a_continuous', model, params, config, 30)
        a = PlanAPolicy('plan_a_balanced', model, params, config, 30)
        for _ in range(150):
            weights = rng.dirichlet([2., 2., 2.])
            c = PolicyContext(4, (), tuple(100*weights), tuple(p.anchor), 1., 100., False, .005)
            da, dp = a.decide(c), p.decide(c)
            def net(decision):
                w = weights if decision.target_weights is None else np.asarray(decision.target_weights)
                v = 1+p.anchor_mean-.5*p.gamma*((w-p.anchor)@p.cov@(w-p.anchor))
                return (1-.005*.5*np.abs(w-weights).sum())*v
            self.assertGreaterEqual(net(dp)+1e-13, net(da))
            q = .5*p.gamma*((weights-p.anchor)@p.cov@(weights-p.anchor))
            f = .005*.5*np.abs(weights-p.anchor).sum()
            l = continuous_fraction(1+p.anchor_mean, q, f)
            grid = np.linspace(0., 1., 10001)
            ref = np.max((1-f*grid)*(1+p.anchor_mean-q*(1-grid)**2))
            value = (1-f*l)*(1+p.anchor_mean-q*(1-l)**2)
            self.assertGreaterEqual(value+1e-13, ref)
        self.assertEqual(continuous_fraction(1., .1, 0.), 1.)
        self.assertEqual(continuous_fraction(1., 0., .01), 0.)

    def test_band_reaches_boundary_only_after_cost_gate(self):
        _, config, model, params = fixture()
        p = make_submission_policy('submission_a_band', model, params, config, 30)
        c = PolicyContext(3, (), (80., 10., 10.), tuple(p.anchor), 1., 100., False, 0.)
        d = p.decide(c)
        self.assertIsNotNone(d.target_weights)
        self.assertAlmostEqual(max(abs(np.asarray(d.target_weights)-p.anchor)), .05)
        self.assertGreater(d.estimated_benefit, d.estimated_cost+p.buffer)
        inside = replace(c, holdings=(51., 29., 20.))
        self.assertEqual(p.decide(inside).action, Action.KEEP)
        self.assertEqual(p.decide(replace(c, real_peak=400.)).action, Action.HUMAN_REVIEW)

    def test_capped_rollouts_match_independent_scalar_accounting(self):
        _, config, model, params = fixture()
        for horizon, name in product((5, 30), ('submission_c_cap10', 'submission_c_cap20')):
            p = make_submission_policy(name, model, params, config, horizon)
            c = PolicyContext(2, (), (80., 10., 10.), tuple(p.anchor), 1., 110., False, .005, .15)
            r = rollout(p, c, 1, 1.)
            for kind in range(4):
                growth, losses = [], []
                for path in p.future:
                    holdings = np.asarray(c.holdings)
                    peak, mdd, pending = c.real_peak, c.historical_max_drawdown, False
                    for step, t in enumerate(range(c.period-1, horizon)):
                        wealth = sum(holdings); current = holdings/wealth
                        ctx = PolicyContext(t+1, (), tuple(holdings), tuple(p.anchor), 1., peak, pending, .005, mdd)
                        a = PlanAPolicy('plan_a_balanced', model, params, config, horizon)
                        d = a.decide(ctx) if not pending else None
                        if d is not None and d.action == Action.HUMAN_REVIEW:
                            pending = True
                        advice = np.asarray(d.target_weights) if d is not None and d.target_weights is not None else current
                        target = advice
                        if step == 0:
                            target = [current, advice, p.anchor, p.defensive][kind]
                            moved = .5*np.abs(target-current).sum()
                            target = current+min(1., p.move_cap/moved)*(target-current) if moved else current
                        if pending:
                            target = current
                        trade = execute_trade(tuple(holdings), tuple(target), .005)
                        holdings = np.asarray(trade.post_trade_holdings)
                        mdd = max(mdd, 1-sum(holdings)/peak); peak = max(peak, sum(holdings))
                        holdings *= 1+path[t]
                        mdd = max(mdd, 1-sum(holdings)/peak); peak = max(peak, sum(holdings))
                    growth.append(np.log(sum(holdings)/sum(c.holdings)))
                    losses.append(-np.log1p(-mdd))
                self.assertAlmostEqual(r['growth'][kind], np.mean(growth), places=12)
                self.assertAlmostEqual(r['risk'][kind], expected_shortfall(losses, .9), places=12)
                self.assertLessEqual(.5*np.abs(r['targets'][kind]-np.asarray(c.holdings)/100).sum(), p.move_cap+1e-12)

    def test_actual_caps_causality_short_group_and_selected_dispatch(self):
        scenarios, config, model, params = fixture()
        s = replace(scenarios[0], scenario_id=100,
                    observations=tuple(replace(o, scenario_id=100) for o in scenarios[0].observations))
        changed = replace(s, observations=s.observations[:3]+tuple(
            replace(o, nominal_returns=(.8, -.5, .01)) for o in s.observations[3:]))
        for name in BASE_RULES[1:]:
            p = make_submission_policy(name, model, params, config, 5)
            r = simulate(s, p, p.initial_target, SimulationConfig(5))
            other = make_submission_policy('submission_selected', model, dict(params, execution_rule=name), config, 5)
            same = simulate(s, other, other.initial_target, SimulationConfig(5))
            self.assertEqual([x.end_holdings for x in r.records], [x.end_holdings for x in same.records])
            other = make_submission_policy(name, model, params, config, 5)
            altered = simulate(changed, other, other.initial_target, SimulationConfig(5))
            self.assertEqual(r.records[:3], altered.records[:3])
            self.assertEqual(r.records[3].target_weights, altered.records[3].target_weights)
            self.assertEqual(r.records[3].action, altered.records[3].action)
            if getattr(p, 'move_cap', None) is not None:
                for record in r.records[1:]:
                    self.assertLessEqual(record.moved_amount/sum(record.pre_trade_holdings), p.move_cap+1e-12)
            if name.startswith('submission_b_'):
                a = PlanAPolicy('plan_a_balanced', model, params, config, 5)
                baseline = simulate(s, a, a.initial_target, SimulationConfig(5))
                self.assertEqual([x.end_holdings for x in r.records], [x.end_holdings for x in baseline.records])

    def test_selector_prefers_simple_and_records_infeasible_fallback(self):
        _, config, _, _ = fixture()
        config['horizons'] = [5, 30]
        rows = [dict(policy=name, scenario_id=s, horizon=h, real_cagr=.03,
                     real_max_drawdown=.1, plan_a_validation_budget=.5,
                     plan_a_risk_budget=.4, annual_growth=.03, review=False)
                for name, s, h in product(BASE_RULES, range(1, 13), (5, 30))]
        scores = pd.DataFrame(rows)
        chosen, trials = choose_rule(scores, config)
        self.assertEqual(chosen['execution_rule'], 'plan_a_balanced')
        self.assertTrue(chosen['execution_selection_feasible'])
        scores['real_max_drawdown'] = .9
        chosen, trials = choose_rule(scores, config)
        self.assertFalse(chosen['execution_selection_feasible'])
        self.assertEqual(len(trials), 8)
        self.assertTrue(all(not t['selection_feasible'] for t in trials))

    def test_customer_stress_schedule_floor_and_risk_order(self):
        _, config, _, _ = fixture()
        config = copy.deepcopy(config)
        config['horizons'] = [5, 6, 15, 25, 30]
        config['submission'] = dict(enabled=True, balanced_policy='submission_selected',
            liquidity_mode='strategic_bill_floor', liquidity_floor=.4, suitability_family_size=440)
        config['customer_stress_costs'] = [.005]
        self.assertEqual(dict(evaluation_schedule(config))[.005], config['horizons'])
        rows = [dict(policy=name, scenario_id=s, horizon=h, cost_rate=cost,
                     real_cagr=growth, real_max_drawdown=risk, strategic_bill_min=.5)
                for (name, growth, risk), s, h, cost in product(
                    [('plan_a_conservative', .01, .01), ('submission_selected', .03, .02), ('plan_a_growth', .04, .03)],
                    range(1, 121), config['horizons'], (.0025, .005))]
        metrics = pd.DataFrame(rows)
        assignments, risk = suitability(metrics, config)
        self.assertEqual(assignments.set_index('profile_id').loc['E', 'portfolio'], 'balanced')
        self.assertTrue(risk.risk_order_ok.all())
        bad = metrics.copy()
        mask = (bad.policy == 'submission_selected')&(bad.horizon == 6)&(bad.cost_rate == .005)
        bad.loc[mask, 'real_max_drawdown'] = .8
        assignments, risk = suitability(bad, config)
        self.assertEqual(assignments.set_index('profile_id').loc['E', 'portfolio'], 'human_review')
        risk = risk[(risk.profile == 'E')&(risk.portfolio == 'balanced')]
        self.assertFalse(risk.iloc[0].cost_stress_ok)
        self.assertFalse(risk.iloc[0].risk_order_ok)

