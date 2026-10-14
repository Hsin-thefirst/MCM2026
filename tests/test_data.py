import csv
import tempfile
import unittest
from pathlib import Path
from safeinvest.data import COLUMNS, load_scenarios, split_scenarios


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test the full-grid legacy loader with generated data, never the real holdout.
        cls.raw_rows = [dict(zip(COLUMNS, (str(sid), str(period), '0.05', '0.02', '0.01', '0.015')))
                        for sid in range(1, 181) for period in range(1, 31)]

    def write_and_load(self, rows, columns=COLUMNS):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            return load_scenarios(path)

    def test_generated_data_and_fixed_split(self):
        scenarios = self.write_and_load(self.raw_rows)
        dev, final = split_scenarios(scenarios)
        self.assertEqual([s.scenario_id for s in dev], list(range(1, 121)))
        self.assertEqual([s.scenario_id for s in final], list(range(121, 181)))
        for scenario in scenarios:
            self.assertEqual([o.period for o in scenario.observations], list(range(1, 31)))
            self.assertTrue(all(o.scenario_id == scenario.scenario_id for o in scenario.observations))
        observation = scenarios[0].observations[0]
        self.assertEqual(observation.nominal_returns, (0.05, 0.02, 0.01))
        self.assertEqual(observation.inflation, 0.015)

    def test_unsorted_input_is_sorted(self):
        scenarios = self.write_and_load(list(reversed(self.raw_rows)))
        self.assertEqual(scenarios[0].scenario_id, 1)
        self.assertEqual(scenarios[-1].observations[-1].period, 30)

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate key'):
            self.write_and_load(self.raw_rows + [self.raw_rows[0]])

    def test_missing_period_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected periods'):
            self.write_and_load(self.raw_rows[1:])

    def test_missing_scenario_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected scenarios'):
            self.write_and_load(self.raw_rows[30:])

    def test_invalid_values_rejected(self):
        for column, value in (('inflation', '-1'), ('equity_return', '-1.1'),
                              ('bond_return', 'nan'), ('bill_proxy_return', 'inf'),
                              ('scenario_id', '181'), ('period', '0')):
            rows = [dict(self.raw_rows[0], **{column: value})] + self.raw_rows[1:]
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.write_and_load(rows)

    def test_changed_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected columns'):
            self.write_and_load(self.raw_rows, tuple(reversed(COLUMNS)))


if __name__ == '__main__':
    unittest.main()
