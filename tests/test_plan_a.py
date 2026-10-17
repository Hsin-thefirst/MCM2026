"""Generated-data checks for convex allocation, causal execution and horizon dispatch."""
from dataclasses import replace
import json
import unittest

import numpy as np

from safeinvest.contracts import Action, MarketObservation, PolicyContext, Scenario, SimulationConfig
from safebench import ROOT
from safebench.engine import simulate
from safebench.experiment import evaluate
from safebench.metrics import compute_metrics
from safebench.plan_a import (PlanAPolicy, annual_statistics, fit_plan_a, log_path_values,
                             solve_allocation)
from safebench.policies import StaticPolicy

CONFIG = json.loads((ROOT / 'configs/benchmark.json').read_text())


def model_fixture():
    entries = {}
    for h, w in [(5, [.2, .3, .5]), (30, [.6, .3, .1])]:
        entries[f'balanced:{h}:1'] = dict(weights=w, review_threshold=.35, risk_budget=.3)
    return dict(train_ids=[1], mu=[.04, .02, .01], cov=np.diag([.04, .01, .001]).tolist(),
                plan_a=dict(train_ids=[1], entries=entries))


class ConvexAllocationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        self.returns = rng.uniform([-.4, -.15, -.02], [.6, .22, .04], (24, 15, 3))

    def test_peak_to_trough_includes_initial_peak_and_recovery(self):
        r = np.array([[[-.2, 0, 0], [.5, 0, 0], [-.1, 0, 0]],
                      [[.1, 0, 0], [-.3, 0, 0], [.2, 0, 0]]])
        v = log_path_values(r, [1, 0, 0], .5)
        np.testing.assert_allclose(v['log_drawdowns'], -np.log([.8, .7]))
        self.assertAlmostEqual(v['risk'], -np.log(.7))

    def test_convexity_and_supporting_gradients(self):
        rng = np.random.default_rng(50)
        w = np.array([.43, .32, .25])
        value = log_path_values(self.returns, w, .83)
        for key in ['risk', 'objective']:
            numerical = []
            for i in range(3):
                delta = np.eye(3)[i] * 1e-6
                numerical.append((log_path_values(self.returns, w + delta, .83)[key] -
                                  log_path_values(self.returns, w - delta, .83)[key]) / 2e-6)
            np.testing.assert_allclose(value[key + '_gradient'], numerical, atol=1e-8)
            for _ in range(15):
                other = rng.dirichlet([1, 1, 1])
                v = log_path_values(self.returns, other, .83)
                middle = log_path_values(self.returns, (w + other) / 2, .83)
                self.assertLessEqual(middle[key], (value[key] + v[key]) / 2 + 1e-12)
                self.assertGreaterEqual(v[key], value[key] + value[key + '_gradient'] @ (other - w) - 1e-12)

    def test_continuous_solution_with_certificate_and_grid_comparison(self):
        budget = log_path_values(self.returns, [.5, .35, .15])['risk']
        fit = solve_allocation(self.returns, budget, bill_floor=.2)
        self.assertLessEqual(fit['risk'], budget + 1e-8)
        self.assertGreaterEqual(fit['weights'][2], .2 - 1e-9)
        self.assertLessEqual(fit['optimality_gap'], 1e-6)
        self.assertGreaterEqual(fit['grid_growth_gain'], -1e-12)
        c = fit['lp_certificate']
        dual = np.dot(c['rhs'], c['inequality_duals']) + c['equality_dual']
        dual += .2 * c['lower_duals'][2] + sum(c['upper_duals'][:3])
        self.assertAlmostEqual(dual, fit['lower_bound'], places=8)

    def test_infeasible_and_unconverged_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'infeasible'):
            solve_allocation(np.full((4, 5, 3), -.1), .01)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            solve_allocation(self.returns, .1, options={'max_iterations': 0})
        with self.assertRaises(PermissionError):
            fit_plan_a([Scenario(121, ())], CONFIG)

    def test_fee_aware_training_statistics_match_engine(self):
        returns = self.returns[:5, :5]
        weight = np.array([.4, .3, .3])
        rows = []
        for sid, path in enumerate(returns):
            observations = tuple(MarketObservation(sid + 1, t + 1, tuple(r), 0.) for t, r in enumerate(path))
            policy = StaticPolicy('annual', tuple(weight), 'annual')
            rows.append(compute_metrics(simulate(Scenario(sid + 1, observations), policy, tuple(weight), SimulationConfig(5))))
        stats = annual_statistics(returns, weight, .0025, .9)
        self.assertAlmostEqual(stats['p90_drawdown'], np.quantile([x['real_max_drawdown'] for x in rows], .9))
        self.assertAlmostEqual(stats['median_growth'], np.median([x['real_cagr'] for x in rows]))


class PlanAExecutionTests(unittest.TestCase):
    def policy(self, name='plan_a_balanced', gamma=4):
        return PlanAPolicy(name, model_fixture(), dict(risk_ratio=1., tracking_gamma=gamma), CONFIG, 30)

    def context(self, holdings=(90., 5., 5.), cost=.0025):
        return PolicyContext(2, (), holdings, (.6, .3, .1), 1., 100., False, cost)

    def test_gate_exact_cost_and_no_trade_at_anchor(self):
        p = self.policy()
        d = p.decide(self.context())
        self.assertIn(d.action, [Action.REBALANCE, Action.CHANGE_TARGET])
        cost = .0025 * np.abs(np.array(d.target_weights) - [.9, .05, .05]).sum() / 2
        self.assertAlmostEqual(d.estimated_cost, cost)
        self.assertGreater(d.estimated_benefit, cost + p.buffer)
        self.assertEqual(p.decide(self.context((60., 30., 10.))).action, Action.KEEP)
        self.assertEqual(p.decide(self.context(cost=.9)).action, Action.KEEP)

    def test_partial_move_can_dominate_full_trade(self):
        p = self.policy(gamma=1.)
        d = p.decide(self.context(cost=.01))
        self.assertEqual(d.action, Action.CHANGE_TARGET)
        self.assertNotEqual(d.target_weights, p.initial_target)
        self.assertGreater(d.estimated_benefit, d.estimated_cost + p.buffer)

    def test_review_and_future_mutation(self):
        self.assertEqual(self.policy().decide(self.context((30., 20., 10.))).action, Action.HUMAN_REVIEW)
        observations = tuple(MarketObservation(2, t, (.1, -.02, .01), .01) for t in range(1, 31))
        s = Scenario(2, observations)
        changed = replace(s, observations=observations[:20] + tuple(replace(o, nominal_returns=(-.8, .8, .4)) for o in observations[20:]))
        a, b = self.policy(), self.policy()
        ra = simulate(s, a, a.initial_target, SimulationConfig(30))
        rb = simulate(changed, b, b.initial_target, SimulationConfig(30))
        self.assertEqual(ra.records[:20], rb.records[:20])
        self.assertEqual(ra.records[20].target_weights, rb.records[20].target_weights)
        self.assertEqual(ra.records[20].action, rb.records[20].action)

    def test_horizon_specific_simulation_and_fixed_anchor(self):
        s = Scenario(2, tuple(MarketObservation(2, t, (0., 0., 0.), 0.) for t in range(1, 31)))
        metrics, records = evaluate([s], model_fixture(), {'plan_a_balanced': dict(risk_ratio=1., tracking_gamma=4.)},
                                   CONFIG, names=['plan_a_balanced'], horizons=[5, 30])
        self.assertEqual(len(records), 35)
        self.assertEqual(set(metrics.simulation_horizon), {5, 30})
        initial = records[records.period == 1].set_index('simulation_horizon')
        self.assertAlmostEqual(initial.loc[5, 'target_weights_equity'], .2)
        self.assertAlmostEqual(initial.loc[30, 'target_weights_equity'], .6)
        with self.assertRaises(ValueError):
            PlanAPolicy('plan_a_balanced', model_fixture(), {}, CONFIG, None)


if __name__ == '__main__':
    unittest.main()
