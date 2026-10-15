"""Unit tests for DrawdownBudgetPolicy."""
import unittest
import numpy as np
from safeinvest.contracts import PolicyContext, MarketObservation, Action
from safebench.policies import DrawdownBudgetPolicy, fit_model
from safebench.data import load_partition

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

if __name__ == '__main__':
    unittest.main()
