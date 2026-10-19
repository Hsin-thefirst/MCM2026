"""Generated-ledger regression checks for the merged report asset builder."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_report", ROOT / "scripts/build_formal_report.py")
report = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(report)
except ModuleNotFoundError as exc:
    if not exc.name.startswith("matplotlib"):
        raise
    report = None


@unittest.skipIf(report is None, "Report plotting requires the optional analysis dependencies")
class FormalReportTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for period in range(1, 31):
            rows.append(dict(policy="plan_a_balanced", scenario_id=7, simulation_horizon=30,
                             cost_rate=.0025, period=period,
                             real_wealth=100 + 8 * period if period <= 6 else 148 - 3.25 * (period - 6),
                             action="human_review" if period == 23 else "keep",
                             moved_amount=1. if period <= 7 else 0., review_pending=period >= 23))
        frame = pd.DataFrame(rows)
        severe = dict(scenario_id=7, peak=148., trough=70., terminal_real_wealth=70.,
                      review_periods=[23], trades=6)
        return frame, severe

    def extract(self, frame, severe):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "generated_ledger.csv"
            frame.to_csv(ledger, index=False)
            return report.read_saved_severe_path(ledger, severe)

    def test_extracts_only_the_requested_recorded_path(self):
        frame, severe = self.fixture()
        noise = pd.concat([frame.assign(policy="equal_annual"), frame.assign(scenario_id=8),
                           frame.assign(simulation_horizon=15), frame.assign(cost_rate=.005)])
        noise["real_wealth"] = 9999.
        result = self.extract(pd.concat([noise, frame.iloc[::-1]]), severe)
        self.assertEqual(result.period.tolist(), list(range(1, 31)))
        self.assertEqual(result.real_wealth.iloc[-1], 70.)
        self.assertEqual(result.loc[result.action == "human_review", "period"].tolist(), [23])

    def test_rejects_incomplete_saved_path(self):
        frame, severe = self.fixture()
        with self.assertRaises(AssertionError):
            self.extract(frame.loc[frame.period != 12], severe)

    def test_rejects_trading_while_review_is_pending(self):
        frame, severe = self.fixture()
        frame.loc[frame.period == 7, "moved_amount"] = 0.
        frame.loc[frame.period == 24, "moved_amount"] = 1.
        with self.assertRaises(AssertionError):
            self.extract(frame, severe)


if __name__ == "__main__":
    unittest.main()
