"""Focused tests for the horizon-dependent budget-value candidate."""
import unittest

import numpy as np

from safeinvest.contracts import Action, MarketObservation, PolicyContext, Scenario, SimulationConfig, Decision
from safebench.engine import simulate
from safebench.policies import BudgetValuePolicy, fit_model, make_policy


CONFIG = {
    'covariance_shrinkage': 0.25,
    'cvar_alpha': 0.9,
    'cvar_limit': 0.08,
    'fixed_gamma': 5.0,
    'review_drawdown': 0.25,
    'periodic_interval': 3,
    'rebalance_threshold': 0.05,
}
PARAMS = {
    'k': 1.0,
    'derisk_threshold': 0.02,
    'gamma': 5.0,
    'buffer': 0.0,
    'risk_penalty': 1.0,
}


def generated_scenarios(count=4, periods=30):
    scenarios = []
    for sid in range(1, count + 1):
        observations = []
        for period in range(1, periods + 1):
            observations.append(MarketObservation(
                sid, period,
                (0.01 + 0.001 * sid + 0.0001 * period,
                 0.004 + 0.0002 * sid,
                 0.002 + 0.0001 * period),
                0.001,
            ))
        scenarios.append(Scenario(sid, tuple(observations)))
    return tuple(scenarios)


def context(equity, peak=100.0, target_equity=None, period=2):
    if target_equity is None:
        target_equity = equity
    target = (target_equity, (1 - target_equity) * .7, (1 - target_equity) * .3)
    holdings = tuple(100 * x for x in (equity, (1 - equity) * .7, (1 - equity) * .3))
    return PolicyContext(period, (), holdings, target, 1.0, peak, False, .0025)


class BudgetValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = generated_scenarios()
        cls.model = fit_model(cls.training, CONFIG)

    def test_fit_model_stores_training_annual_real_rows(self):
        expected = np.array([
            [(1 + r) / (1 + observation.inflation) - 1
             for r in observation.nominal_returns]
            for scenario in self.training for observation in scenario.observations
        ])
        np.testing.assert_allclose(np.asarray(self.model['risk_rows']), expected)
        self.assertEqual(len(self.model['risk_rows']), 4 * 30)

    def _risk_model(self):
        # Hand-calculable annual training returns isolate the shortfall term.
        return dict(self.model, mu=[.08, 0., 0.],
                    cov=np.diag([.001, .0004, .0001]).tolist(),
                    risk_rows=[[-.8, 0., 0.], [.96, 0., 0.]])

    def test_value_benefit_matches_training_risk_formula(self):
        model = self._risk_model()
        policy = make_policy('budget_lifecycle_balanced', model, PARAMS, CONFIG, horizon=30)
        # dd=10% => target equity=.15+(.60-.15)*.6=.42.
        c = context(.60, peak=100 / .9, target_equity=.60)
        decision = policy.decide(c)
        current = np.array([.60, .28, .12])
        target = np.array([.42, .406, .174])
        def expected_value(weights):
            annual_losses = [max(0., 1 - .9 * (1 + sum(r * w for r, w in zip(row, weights))) - .25)
                             for row in model['risk_rows']]
            variance = sum(v * w ** 2 for v, w in zip([.001, .0004, .0001], weights))
            return 1 + .08 * weights[0] - 5 / 2 * variance - sum(annual_losses) / 2
        target_value = expected_value(target)
        current_value = expected_value(current)
        self.assertAlmostEqual(policy._utility(target, .1), target_value)
        self.assertAlmostEqual(policy._utility(current, .1), current_value)
        expected_benefit = (target_value - current_value) / target_value
        expected_cost = c.transaction_cost_rate * .5 * np.abs(target - current).sum()
        self.assertAlmostEqual(decision.estimated_benefit, expected_benefit)
        self.assertAlmostEqual(decision.estimated_cost, expected_cost)
        self.assertEqual(decision.action, Action.CHANGE_TARGET)
        self.assertGreater(expected_benefit, expected_cost)
        self.assertFalse(decision.gate_exempt)

    def test_large_budget_contraction_still_uses_complete_gate(self):
        # Equity's training return is positive, so a large forced sell-back has
        # negative value benefit; it must not be silently exempted.
        policy = make_policy('budget_lifecycle_balanced', self.model,
                             dict(PARAMS, derisk_threshold=0.02), CONFIG, horizon=30)
        decision = policy.decide(context(.60, peak=125.0, target_equity=.60))
        self.assertEqual(decision.action, Action.KEEP)
        self.assertIn('BENEFIT_BELOW_COST', decision.reason_codes)
        self.assertFalse(decision.gate_exempt)
        self.assertLessEqual(decision.estimated_benefit, decision.estimated_cost)

    def test_small_equity_drift_uses_band_only_when_budget_target_is_stable(self):
        policy = make_policy('budget_lifecycle_balanced', self.model, PARAMS, CONFIG, horizon=30)
        # At 10% drawdown the budget target is 42%; current equity is only 1pp above it.
        decision = policy.decide(context(.43, peak=100 / .9, target_equity=.42))
        self.assertEqual(decision.action, Action.KEEP)
        self.assertIn('BUDGET_DRIFT_BAND', decision.reason_codes)
        # A budget contraction larger than the band cannot be hidden by drift suppression.
        decision = policy.decide(context(.43, peak=100 / .6, target_equity=.42))
        self.assertNotIn('BUDGET_DRIFT_BAND', decision.reason_codes)

    def test_zero_band_forces_zero_threshold(self):
        policy = make_policy('budget_lifecycle_balanced_zero_band', self.model,
                             PARAMS, CONFIG, horizon=30)
        self.assertEqual(policy.derisk_threshold, 0.0)
        decision = policy.decide(context(.43, peak=100 / .9, target_equity=.42))
        self.assertNotIn('BUDGET_DRIFT_BAND', decision.reason_codes)

    def test_static_and_diagnostic_siblings_share_initial_target(self):
        names = [
            'budget_lifecycle_balanced',
            'budget_lifecycle_balanced_zero_band',
            'budget_lifecycle_balanced_static',
            'budget_lifecycle_balanced_annual',
            'budget_lifecycle_balanced_hold',
        ]
        policies = [make_policy(name, self.model, PARAMS, CONFIG, horizon=15) for name in names]
        for policy in policies[1:]:
            np.testing.assert_allclose(policy.initial_target, policies[0].initial_target)
        self.assertEqual(policies[2].initial_target, policies[3].weights)
        self.assertEqual(policies[2].initial_target, policies[4].weights)

    def test_policy_is_horizon_dependent(self):
        short = make_policy('budget_lifecycle_balanced', self.model, PARAMS, CONFIG, horizon=5)
        long = make_policy('budget_lifecycle_balanced', self.model, PARAMS, CONFIG, horizon=30)
        self.assertLess(short.initial_target[0], long.initial_target[0])

    def test_invalid_threshold_rejected(self):
        for value in (-0.01, 1.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                make_policy('budget_lifecycle_balanced', self.model,
                            dict(PARAMS, derisk_threshold=value), CONFIG)

    def test_engine_rejects_non_decreasing_exempt_trade(self):
        class BadExemption:
            name = 'bad_exemption'
            requires_benefit_gate = True
            buffer = 0.0

            def decide(self, c):
                return Decision(Action.CHANGE_TARGET, ('DRAWDOWN_RISK_REDUCTION',), 'bad',
                                (.5, .35, .15), -1.0, 0.0, gate_exempt=True)

        scenario = generated_scenarios(1, 2)[0]
        with self.assertRaisesRegex(ValueError, 'strict equity reduction'):
            simulate(scenario, BadExemption(), (.333333333333, .333333333333, .333333333334),
                     SimulationConfig(2, 100, 0.0))


if __name__ == '__main__':
    unittest.main()
