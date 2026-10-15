"""Generated-data regression tests for budget selection artifacts and assignments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from safebench import ROOT
from safebench.data import load_partition
from safebench.experiment import (
    assignment_policy, code_digest, digest, freeze, manifest, run_final, verify_bundle,
)


class BudgetFreezeTests(unittest.TestCase):
    def test_digest_rejects_nonfinite_instead_of_aliasing_null(self):
        self.assertIsInstance(digest({'gap': None}), str)
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                digest({'gap': value})

    def test_assignment_uses_exact_policy_and_accepts_legacy_portfolio(self):
        self.assertEqual(assignment_policy({'portfolio': 'balanced',
                         'policy': 'budget_lifecycle_balanced'}), 'budget_lifecycle_balanced')
        self.assertEqual(assignment_policy({'portfolio': 'conservative'}), 'conservative_annual')
        self.assertEqual(assignment_policy({'portfolio': 'human_review'}), 'human_review')

    def test_fallback_freeze_and_dynamic_final_assignment_on_generated_data(self):
        config = json.loads((ROOT / 'configs/benchmark.json').read_text())
        config = dict(config, horizons=[5, 30], sensitivity_costs=[.005], bootstrap=20)
        policy = 'budget_lifecycle_balanced'
        params = {policy: {'by_horizon': {
            str(h): dict(config['budget_value_tuning_grid'][0]) for h in config['horizons']
        }}}
        trials = [dict(policy=policy, horizon=h, selected_trial=0,
                       constraint_fallback=True, paired_gap_se=float('nan'))
                  for h in config['horizons']]
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark') as directory:
            root = Path(directory)
            data = root / 'generated.csv'
            with data.open('w') as stream:
                stream.write('scenario_id,period,equity_return,bond_return,bill_proxy_return,inflation\n')
                for sid in range(1, 181):
                    for period in range(1, 31):
                        stream.write(f'{sid},{period},0,0,0,0\n')
            dev = root / 'dev'
            dev.mkdir()
            _, sha = load_partition(data)
            (dev / 'manifest.json').write_text(json.dumps(
                manifest(config, sha, list(range(1, 121)), 'development')))
            pd.DataFrame([
                dict(profile_id='dynamic', horizon=5, loss_tolerance=.25,
                     portfolio='balanced', policy=policy),
                dict(profile_id='review', horizon=5, loss_tolerance=.1,
                     portfolio='human_review', policy='human_review'),
            ]).to_csv(dev / 'suitability.csv', index=False)
            bundle = root / 'bundle.json'
            with patch('safebench.experiment.tune', return_value=(params, trials, [])):
                freeze(config, data, 'balanced_hold', bundle, dev)
            raw = json.loads(bundle.read_text())
            self.assertIsNone(raw['tuning_trials'][0]['paired_gap_se'])
            self.assertEqual(raw['selection_diagnostics']['status'], 'research_only_if_fallback')
            self.assertFalse(raw['selection_diagnostics']['deployment_established'])
            verify_bundle(raw)
            frozen_hash = code_digest()
            with patch('safebench.experiment.BASE', root), \
                 patch('safebench.experiment.code_digest', return_value=frozen_hash), \
                 patch('safebench.report.generate_report'):
                run_final(bundle, data, root / 'final', True)
            metrics = pd.read_csv(root / 'final/scenario_metrics.csv')
            assignments = pd.read_csv(root / 'final/suitability.csv').set_index('profile_id')
            dynamic = assignments.loc['dynamic']
            selected = metrics[(metrics.policy == policy) & (metrics.horizon == 5)
                               & (metrics.cost_rate == config['cost_rate'])]
            self.assertEqual(len(selected), 60)
            self.assertEqual(dynamic.policy, policy)
            self.assertAlmostEqual(dynamic.final_median_growth, selected.real_cagr.median())
            self.assertAlmostEqual(dynamic.final_breach_probability,
                                   (selected.real_max_drawdown > .25).mean())
            self.assertNotIn('human_review', set(metrics.policy))
            self.assertTrue(json.loads((root / 'final/validation.json').read_text())['all_checks_passed'])


if __name__ == '__main__':
    unittest.main()
