"""Synthetic validation of the planning probes, without official holdout data."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_predictability import synthetic_self_check, design, fit_ridge
from probe_strategy_routes import check_static, ProbeGate
from safebench.engine import simulate
from safebench.policies import grid, moments
from safebench.data import real_rows
from safeinvest.contracts import MarketObservation, Scenario, SimulationConfig


class PlanningProbeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(919)
        self.scenarios = tuple(Scenario(sid, tuple(MarketObservation(
            sid, t + 1, tuple(rng.uniform(-.25, .3, 3)), float(rng.uniform(-.01, .04)))
            for t in range(8))) for sid in [1, 2, 3, 4])
        self.config = dict(static_modes=["annual", "hold", "band_full", "band_edge"],
                           band=.05, buffer=.0001, drawdown_response=2., drawdown_scale=.4)

    def test_predictor_prefix(self):
        self.assertTrue(synthetic_self_check()["passed"])

    def test_vector_accounting_against_scalar_engine(self):
        check_static(self.scenarios, np.asarray([[1/3]*3, [.5, .35, .15], [0., 0., 1.]]),
                     .005, self.config, True)

    def test_gate_actions_ignore_future(self):
        training = self.scenarios[:3]
        mu, cov = moments(real_rows(training))
        x, y, _ = design(training)
        predictor = fit_ridge(x, y)
        original = self.scenarios[3]
        altered = replace(original, observations=original.observations[:4] + tuple(
            replace(o, nominal_returns=(.8, -.4, .2), inflation=.15)
            for o in original.observations[4:]))
        for mode in ["constant", "ridge_half", "drawdown"]:
            def run(s):
                policy = ProbeGate(mode, mu, cov, predictor, grid(((0., 1.),)*3), 5., mode, self.config)
                return simulate(s, policy, policy.initial_target, SimulationConfig(8, 100., .0025))
            a, b = run(original), run(altered)
            self.assertEqual(a.records[:4], b.records[:4])
            self.assertEqual(a.records[4].action, b.records[4].action)
            self.assertEqual(a.records[4].target_weights, b.records[4].target_weights)


if __name__ == "__main__":
    unittest.main()
