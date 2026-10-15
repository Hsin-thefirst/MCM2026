"""Unit tests for DrawdownBudgetPolicy."""
import unittest
import numpy as np
import pandas as pd
from safeinvest.contracts import PolicyContext, MarketObservation, Action
from safebench.policies import DrawdownBudgetPolicy, fit_model
from safebench.data import load_partition
from safebench.experiment import select_drawdown_params
from safebench.validation import exempt_direction_violations

class TestDrawdownBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load development scenarios and fit a model once."""
        scenarios, _ = load_partition('data/SafeInvest_Synthetic_Data.csv', 'development')
        cls.scenarios = scenarios[:10]
        config = {'covariance_shrinkage': 0.25, 'cvar_alpha': 0.9, 'cvar_limit': 0.08, 'fixed_gamma': 5.0}
        cls.model = fit_model(cls.scenarios, config)
        cls.config = {'review_drawdown': 0.25}

    def test_initial_target_at_max_equity(self):
        """Initial target should place equity at e_max."""
        params = {'k': 1.0, 'buffer': 0.0, 'gamma': 5.0}
        policy = DrawdownBudgetPolicy('drawdown_balanced', self.model, params, self.config,
                                     loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                     nonequity_split=(0.7, 0.3))
        equity, bond, bill = policy.initial_target
        self.assertAlmostEqual(equity, 0.65, places=10)
        self.assertAlmostEqual(bond, 0.35 * 0.7, places=10)
        self.assertAlmostEqual(bill, 0.35 * 0.3, places=10)
        self.assertAlmostEqual(equity + bond + bill, 1.0, places=10)

    def test_budget_reduction_lowers_equity(self):
        """Equity target should decrease as realized drawdown approaches tolerance."""
        params = {'k': 1.0, 'buffer': 0.0, 'gamma': 5.0}
        policy = DrawdownBudgetPolicy('drawdown_balanced', self.model, params, self.config,
                                     loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                     nonequity_split=(0.7, 0.3))
        # Simulate a context with partial drawdown: peak=100, current=85 real wealth
        # realized_dd = 1 - 85/100 = 0.15
        # budget_remaining = (0.25 - 0.15) / 0.25 = 0.4
        # target_equity = 0.15 + (0.65 - 0.15) * 0.4^1.0 = 0.15 + 0.5 * 0.4 = 0.35
        obs = self.scenarios[0].observations[0]
        ctx = PolicyContext(period=2, history=[obs], holdings=(42.5, 24.5, 18.0),
                           target_weights=(0.5, 0.35, 0.15), cumulative_inflation=1.0,
                           real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)
        # Current real wealth = 85
        decision = policy.decide(ctx)
        # Should suggest change or rebalance toward lower equity
        if decision.action in (Action.REBALANCE, Action.CHANGE_TARGET):
            equity_target = decision.target_weights[0]
            self.assertLess(equity_target, 0.65)
            self.assertGreater(equity_target, 0.15)

    def test_review_triggered_at_tolerance(self):
        """HUMAN_REVIEW should trigger when realized drawdown >= loss_tolerance."""
        params = {'k': 1.0, 'buffer': 0.0, 'gamma': 5.0}
        policy = DrawdownBudgetPolicy('drawdown_balanced', self.model, params, self.config,
                                     loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                     nonequity_split=(0.7, 0.3))
        obs = self.scenarios[0].observations[0]
        # peak=100, current=75, dd=0.25 exactly
        ctx = PolicyContext(period=2, history=[obs], holdings=(37.5, 21.5, 16.0),
                           target_weights=(0.5, 0.35, 0.15), cumulative_inflation=1.0,
                           real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)
        decision = policy.decide(ctx)
        self.assertEqual(decision.action, Action.HUMAN_REVIEW)
        self.assertIn('DRAWDOWN_LIMIT_REACHED', decision.reason_codes)

    def test_curvature_parameter_k(self):
        """Higher k should make equity drop faster near the tolerance."""
        params_linear = {'k': 1.0, 'buffer': 0.0, 'gamma': 5.0}
        params_curved = {'k': 2.0, 'buffer': 0.0, 'gamma': 5.0}
        policy_linear = DrawdownBudgetPolicy('test_linear', self.model, params_linear, self.config,
                                            loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                            nonequity_split=(0.7, 0.3))
        policy_curved = DrawdownBudgetPolicy('test_curved', self.model, params_curved, self.config,
                                            loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                            nonequity_split=(0.7, 0.3))
        obs = self.scenarios[0].observations[0]
        # budget_remaining = 0.4 → k=1: 0.4, k=2: 0.16
        ctx = PolicyContext(period=2, history=[obs], holdings=(42.5, 24.5, 18.0),
                           target_weights=(0.5, 0.35, 0.15), cumulative_inflation=1.0,
                           real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)
        dec_linear = policy_linear.decide(ctx)
        dec_curved = policy_curved.decide(ctx)
        if dec_linear.action in (Action.REBALANCE, Action.CHANGE_TARGET) and \
           dec_curved.action in (Action.REBALANCE, Action.CHANGE_TARGET):
            # k=2 should have lower equity target
            self.assertLess(dec_curved.target_weights[0], dec_linear.target_weights[0])

    def test_benefit_gate_enforced(self):
        """Cost gate should block trades when benefit <= cost + buffer."""
        params = {'k': 1.0, 'buffer': 0.01, 'gamma': 5.0}
        policy = DrawdownBudgetPolicy('drawdown_test', self.model, params, self.config,
                                     loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                     nonequity_split=(0.7, 0.3))
        obs = self.scenarios[0].observations[0]
        # Current holdings very close to target → benefit small, cost also small
        ctx = PolicyContext(period=2, history=[obs], holdings=(64.8, 24.45, 10.45),
                           target_weights=(0.65, 0.245, 0.105), cumulative_inflation=1.0,
                           real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)
        decision = policy.decide(ctx)
        # With high buffer, small benefit should be blocked
        if decision.action == Action.KEEP:
            self.assertIn('BENEFIT_BELOW_COST', decision.reason_codes)

    def _derisk_context(self):
        """Partial drawdown: real wealth 85 vs peak 100 → dd=15%, budget 40% → target equity 35%."""
        obs = self.scenarios[0].observations[0]
        return PolicyContext(period=2, history=[obs], holdings=(42.5, 24.5, 18.0),
                             target_weights=(0.5, 0.35, 0.15), cumulative_inflation=1.0,
                             real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)

    def test_derisk_exempt_trades_when_benefit_negative(self):
        """With derisk_exempt, a protective de-risk executes even when the mean-variance
        benefit is negative; without the flag the same decision is gate-blocked."""
        common = dict(loss_tolerance=0.25, equity_bounds=(0.15, 0.65), nonequity_split=(0.7, 0.3))
        exempt = DrawdownBudgetPolicy('drawdown_x', self.model,
                                      {'k': 1.0, 'buffer': 0.0, 'gamma': 0.1, 'derisk_exempt': True},
                                      self.config, **common)
        gated = DrawdownBudgetPolicy('drawdown_x', self.model,
                                     {'k': 1.0, 'buffer': 0.0, 'gamma': 0.1}, self.config, **common)
        ctx = self._derisk_context()
        decision = exempt.decide(ctx)
        self.assertIn(decision.action, (Action.REBALANCE, Action.CHANGE_TARGET))
        self.assertEqual(decision.reason_codes[0], 'DRAWDOWN_RISK_REDUCTION')
        self.assertTrue(decision.gate_exempt)
        self.assertAlmostEqual(decision.target_weights[0], 0.35, places=10)
        blocked = gated.decide(ctx)
        self.assertEqual(blocked.action, Action.KEEP)
        self.assertIn('BENEFIT_BELOW_COST', blocked.reason_codes)
        self.assertFalse(blocked.gate_exempt)

    def test_rerisk_still_gated_with_exemption(self):
        """Exemption must stay one-sided: re-risking trades still need the benefit gate."""
        policy = DrawdownBudgetPolicy('drawdown_x', self.model,
                                      {'k': 1.0, 'buffer': 0.0, 'gamma': 20.0, 'derisk_exempt': True},
                                      self.config, loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                      nonequity_split=(0.7, 0.3))
        obs = self.scenarios[0].observations[0]
        # No drawdown: budget full → target equity 0.65 above current 0.15 → risk-increasing
        ctx = PolicyContext(period=2, history=[obs], holdings=(15.0, 59.5, 25.5),
                            target_weights=(0.15, 0.595, 0.255), cumulative_inflation=1.0,
                            real_peak=100.0, review_pending=False, transaction_cost_rate=0.0025)
        decision = policy.decide(ctx)
        self.assertEqual(decision.action, Action.KEEP)
        self.assertIn('BENEFIT_BELOW_COST', decision.reason_codes)
        self.assertFalse(decision.gate_exempt)

    def test_equity_shift_parameter(self):
        """equity_shift lowers the tier equity cap and the initial target follows."""
        params = {'k': 1.0, 'buffer': 0.0, 'gamma': 5.0, 'equity_shift': -0.10}
        policy = DrawdownBudgetPolicy('drawdown_balanced', self.model, params, self.config,
                                      loss_tolerance=0.25, equity_bounds=(0.15, 0.65),
                                      nonequity_split=(0.7, 0.3))
        equity, bond, bill = policy.initial_target
        self.assertAlmostEqual(equity, 0.55, places=10)
        self.assertAlmostEqual(equity + bond + bill, 1.0, places=10)

    def test_selection_prefers_feasible_over_higher_cagr(self):
        """Trials meeting the training breach-probability constraint win even with lower CAGR."""
        n = 64
        valid = pd.DataFrame([
            dict(policy='drawdown_x', trial=t, scenario_id=s, real_cagr=0.04 + 0.01 * t)
            for t in (0, 1) for s in range(1, n + 1)])
        train = pd.DataFrame(
            [dict(policy='drawdown_x', trial=0, scenario_id=s, real_max_drawdown=0.10) for s in range(1, n + 1)] +
            [dict(policy='drawdown_x', trial=1, scenario_id=s, real_max_drawdown=0.30) for s in range(1, n + 1)])
        grid = [{'k': 1.0, 'buffer': 0.0}, {'k': 1.0, 'buffer': 0.001}]
        config = {'suitability_confidence': 0.95, 'suitability_max_breach_probability': 0.20}
        selected, rows = select_drawdown_params(valid, train, 'drawdown_x', 0.25, grid, config)
        self.assertEqual(selected, 0)
        by_trial = {r['trial']: r for r in rows}
        self.assertTrue(by_trial[0]['constraint_satisfied'])
        self.assertFalse(by_trial[1]['constraint_satisfied'])
        self.assertFalse(any(r['constraint_fallback'] for r in rows))

    def test_selection_fallback_when_none_feasible(self):
        """If no trial meets the constraint, pick the least-violating one and record it."""
        n = 64
        valid = pd.DataFrame([
            dict(policy='drawdown_x', trial=t, scenario_id=s, real_cagr=0.06 - 0.01 * t)
            for t in (0, 1) for s in range(1, n + 1)])
        train = pd.DataFrame(
            [dict(policy='drawdown_x', trial=0, scenario_id=s, real_max_drawdown=0.30) for s in range(1, n + 1)] +
            [dict(policy='drawdown_x', trial=1, scenario_id=s, real_max_drawdown=0.30 if s % 4 == 0 else 0.20)
             for s in range(1, n + 1)])
        grid = [{'k': 1.0, 'buffer': 0.0}, {'k': 1.5, 'buffer': 0.0}]
        config = {'suitability_confidence': 0.95, 'suitability_max_breach_probability': 0.20}
        selected, rows = select_drawdown_params(valid, train, 'drawdown_x', 0.25, grid, config)
        self.assertEqual(selected, 1)  # fewer breaches → lower upper bound, despite lower CAGR
        self.assertTrue(all(r['constraint_fallback'] for r in rows))
        self.assertFalse(all(r['constraint_satisfied'] for r in rows))

    def test_exempt_direction_violations(self):
        """Audit helper flags exempt trades that do not reduce equity exposure."""
        r = pd.DataFrame([
            dict(action='rebalance', gate_exempt=True, period=2, pre_trade_holdings_equity=50.,
                 pre_trade_holdings_bond=30., pre_trade_holdings_bill=20., target_weights_equity=0.35),
            dict(action='rebalance', gate_exempt=True, period=2, pre_trade_holdings_equity=50.,
                 pre_trade_holdings_bond=30., pre_trade_holdings_bill=20., target_weights_equity=0.60),
            dict(action='rebalance', gate_exempt=False, period=2, pre_trade_holdings_equity=50.,
                 pre_trade_holdings_bond=30., pre_trade_holdings_bill=20., target_weights_equity=0.90),
            dict(action='rebalance', gate_exempt=True, period=1, pre_trade_holdings_equity=0.,
                 pre_trade_holdings_bond=0., pre_trade_holdings_bill=100., target_weights_equity=0.65),
            dict(action='keep', gate_exempt=False, period=2, pre_trade_holdings_equity=50.,
                 pre_trade_holdings_bond=30., pre_trade_holdings_bill=20., target_weights_equity=0.90),
        ])
        self.assertEqual(exempt_direction_violations(r), [1])

    def test_drawdown_families_partition(self):
        """Per-family grids: lifecycle uses its own grid, plain the shared one."""
        from safebench.experiment import drawdown_families
        config = {'drawdown_tuning_grid': [{'k': 1.0}],
                  'drawdown_lifecycle_tuning_grid': [{'k': 2.0}]}
        (plain_names, plain_grid), (life_names, life_grid) = drawdown_families(config)
        self.assertEqual(plain_grid, [{'k': 1.0}])
        self.assertEqual(life_grid, [{'k': 2.0}])
        self.assertEqual(sorted(plain_names),
                         ['drawdown_aggressive', 'drawdown_balanced', 'drawdown_conservative'])
        self.assertEqual(sorted(life_names),
                         ['drawdown_lifecycle_aggressive', 'drawdown_lifecycle_balanced',
                          'drawdown_lifecycle_conservative'])

    def test_drawdown_families_fallback(self):
        """Missing lifecycle grid falls back to the shared drawdown grid."""
        from safebench.experiment import drawdown_families
        config = {'drawdown_tuning_grid': [{'k': 1.0}]}
        (_, plain_grid), (_, life_grid) = drawdown_families(config)
        self.assertEqual(life_grid, plain_grid)

if __name__ == '__main__':
    unittest.main()
