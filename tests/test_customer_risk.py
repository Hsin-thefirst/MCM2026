"""Generated-fixture tests for defensive allocations and risk impossibility bounds."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from customer_risk import CustomerRiskPolicy, select_fraction, unavoidable_drawdown
from safeinvest.contracts import SimulationConfig
from safebench.engine import simulate
from safebench.plan_a import PlanAPolicy
from test_plan_bc import fixture


class CustomerRiskTests(unittest.TestCase):
    def test_pathwise_impossibility_bound_for_all_two_period_weights(self):
        returns = np.array([[[-.3, -.2, -.1], [-.4, -.2, -.15]],
                            [[.5, .05, -.02], [-.2, .1, .01]]])
        bound = unavoidable_drawdown(returns)
        self.assertAlmostEqual(bound[0], 1 - .9 * .85)
        self.assertAlmostEqual(bound[1], 0.)
        weights = [(i / 4, j / 4, 1 - (i + j) / 4) for i in range(5) for j in range(5 - i)]
        for first, second in itertools.product(weights, repeat=2):
            wealth = np.ones(2)
            peak = wealth.copy()
            mdd = np.zeros(2)
            for t, target in enumerate([first, second]):
                wealth *= 1 + returns[:, t] @ target
                peak = np.maximum(peak, wealth)
                mdd = np.maximum(mdd, 1 - wealth / peak)
            self.assertTrue(np.all(mdd >= bound - 1e-14))
        with self.assertRaises(ValueError):
            unavoidable_drawdown(np.array([[[-1., 0., 0.]]]))

    def test_identity_causal_prefix_and_pending_review(self):
        scenarios, config, model, params = fixture()
        model["minvar"] = [0., 0., 1.]
        profile = dict(profile_id="E", horizon=5, loss_tolerance=.35, liquidity="high")
        s = replace(scenarios[0], scenario_id=100,
                    observations=tuple(replace(o, scenario_id=100) for o in scenarios[0].observations))
        p = CustomerRiskPolicy("new", model, params, config, profile, 1.)
        old = PlanAPolicy("plan_a_balanced", model, params, config, 5)
        a = simulate(s, p, p.initial_target, SimulationConfig(5))
        b = simulate(s, old, old.initial_target, SimulationConfig(5))
        self.assertEqual(a.records, b.records)
        changed = replace(s, observations=s.observations[:2] + tuple(
            replace(o, nominal_returns=(-.6, -.2, -.1), inflation=.1) for o in s.observations[2:]))
        p = CustomerRiskPolicy("new", model, params, config, profile, .5, early_review=True)
        q = CustomerRiskPolicy("new", model, params, config, profile, .5, early_review=True)
        a = simulate(s, p, p.initial_target, SimulationConfig(5))
        b = simulate(changed, q, q.initial_target, SimulationConfig(5))
        for x, y in zip(a.records[:3], b.records[:3]):
            self.assertEqual(x.post_trade_holdings, y.post_trade_holdings)
        pending = False
        for record in b.records:
            if pending:
                self.assertEqual(record.action.value, "keep")
                self.assertEqual(record.moved_amount, 0.)
            pending = record.review_pending
        self.assertTrue(pending)
        self.assertAlmostEqual(q.review_threshold, .8 * .35)
        with self.assertRaises(PermissionError):
            CustomerRiskPolicy("bad", dict(model, train_ids=[121]), params, config, profile, .5)

    def test_selection_retains_failure_and_rejects_missing_or_duplicate_pairs(self):
        settings = json.loads((ROOT / "configs/customer_risk_review.json").read_text())
        profile = dict(profile_id="B", horizon=15, loss_tolerance=.25, liquidity="medium")
        rows = [dict(fraction=f, cost_rate=c, scenario_id=i, real_max_drawdown=.1 + .15 * f,
                     real_cagr=.01 + .02 * f, strategic_bill_min=.8 - .4 * f)
                for f in settings["mix_fractions"] for c in settings["costs"] for i in range(1, 97)]
        frame = pd.DataFrame(rows)
        chosen = select_fraction(frame, profile, settings)
        self.assertTrue(chosen["feasible"])
        self.assertEqual(chosen["fraction"], .5)
        bad = frame.assign(real_max_drawdown=.5)
        chosen = select_fraction(bad, profile, settings)
        self.assertFalse(chosen["feasible"])
        self.assertEqual(len(chosen["evidence"]), 5)
        with self.assertRaises(ValueError):
            select_fraction(frame.iloc[1:], profile, settings)
        with self.assertRaises(ValueError):
            select_fraction(pd.concat([frame, frame.iloc[:1]]), profile, settings)
