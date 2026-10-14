"""Generated-fixture tests of constrained selection and its paired research boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from safebench import ROOT
from safebench.experiment import digest, tune, freeze, manifest
from safebench.research import compare_selection_runs
from safebench.selection import select_trials, validate_selection_config
from safeinvest.contracts import Scenario, MarketObservation

CONFIG = json.loads((ROOT / 'configs/path_risk.json').read_text())


def fixtures(trials=((.10, .60), (.04, .30))):
    rows = [dict(policy='cost_aware', scenario_id=sid, trial=i, real_cagr=growth, real_max_drawdown=loss)
            for i, (growth, loss) in enumerate(trials) for sid in range(1, 11)]
    reference = pd.DataFrame([dict(policy='balanced_annual', scenario_id=sid,
        real_cagr=.035, real_max_drawdown=.30) for sid in range(1, 11)])
    config = dict(CONFIG, tuning_grid=CONFIG['tuning_grid'][:len(trials)])
    return pd.DataFrame(rows), reference, config


class SelectionTests(unittest.TestCase):
    def test_best_utility_can_fail_path_constraint(self):
        scores, ref, config = fixtures()
        control, _ = select_trials(scores, dict(config, selection_rule='one_se_utility'))
        selected, rows = select_trials(scores, config, ref)
        self.assertEqual(control['cost_aware']['buffer'], 0)
        self.assertEqual(selected['cost_aware']['selection_trial'], 1)
        self.assertFalse(rows[0]['risk_feasible'])
        self.assertTrue(rows[1]['risk_feasible'])

    def test_lower_tail_growth_is_checked_separately(self):
        scores, ref, config = fixtures(((.08, .29), (.04, .30)))
        scores.loc[(scores.trial == 0) & (scores.scenario_id <= 2), 'real_cagr'] = 0
        selected, rows = select_trials(scores, config, ref)
        self.assertFalse(rows[0]['risk_feasible'])
        self.assertEqual(selected['cost_aware']['selection_trial'], 1)

    def test_no_feasible_candidate_is_visible_diagnostic(self):
        scores, ref, config = fixtures(((.10, .60), (.06, .40)))
        selected, rows = select_trials(scores, config, ref)
        self.assertEqual(selected['cost_aware']['selection_trial'], 1)
        self.assertEqual(selected['cost_aware']['selection_status'], 'infeasible_diagnostic')
        self.assertTrue(all(not row['risk_feasible'] for row in rows))

    def test_legacy_one_se_tie_preserves_buffer_then_gamma(self):
        scores, _, config = fixtures(((.04, .30), (.04, .30), (.04, .30), (.04, .30)))
        chosen, _ = select_trials(scores, dict(config, selection_rule='one_se_utility'))
        self.assertEqual(chosen['cost_aware'], {'gamma': 8.0, 'buffer': .001, 'window': 5})

    def test_missing_pairs_are_rejected(self):
        scores, ref, config = fixtures()
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            select_trials(scores.iloc[:-1], config, ref)
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            select_trials(scores, config, ref.iloc[:-1])

    def test_bad_rule_and_invalid_margins_are_rejected(self):
        for changes in ({'selection_rule':'typo'}, {'selection_max_p90_increase':float('nan')},
                        {'selection_reference':'equal_annual'}, {'selection_infeasible_action':'silent'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_selection_config(dict(CONFIG, **changes))

    def test_tune_trace_and_nested_training_isolation(self):
        scenarios = tuple(Scenario(sid, tuple(MarketObservation(sid, year, (0., 0., 0.), 0.)
            for year in range(1, 31))) for sid in range(1, 13))
        config = dict(CONFIG, inner_folds=2, tuning_grid=CONFIG['tuning_grid'][:2])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'trace.csv.gz'
            chosen, trials, splits = tune(scenarios, config, 12, trace_path=path)
            trace = pd.read_csv(path)
            self.assertEqual(len(trace), 12 * (4 * 2 + 1))
            self.assertEqual(len(trials), 8)
            self.assertTrue(all(row['risk_feasible'] for row in trials if row['selected']))
            self.assertTrue(all(value['selection_status'] == 'feasible' for value in chosen.values()))
            for split in splits:
                self.assertTrue(set(split['train_ids']).isdisjoint(split['validation_ids']))
                self.assertEqual(set(trace[trace.fold == split['inner_fold']].scenario_id),
                                 set(split['validation_ids']))
            with self.assertRaises(FileExistsError):
                tune(scenarios, config, 12, trace_path=path)

    def test_infeasible_outer_diagnostic_cannot_be_frozen(self):
        config = dict(CONFIG, outer_folds=1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'manifest.json').write_text(json.dumps(manifest(config, 'fixture', list(range(1, 13)), 'development')))
            pd.DataFrame([dict(policy='cost_aware', selected=True, risk_feasible=False)]).to_csv(
                root / 'tuning_trials.csv', index=False)
            with patch('safebench.experiment.load_partition', side_effect=AssertionError('Unexpected market read')):
                with self.assertRaisesRegex(ValueError, 'infeasible'):
                    freeze(config, root / 'no_market.csv', 'cost_aware', root / 'frozen.json', root)
            self.assertFalse((root / 'frozen.json').exists())


class ResearchComparisonTests(unittest.TestCase):
    def test_generated_chinese_report_is_utf8_with_all_plot_assets(self):
        from safebench.experiment import evaluate
        from safebench.metrics import summarize, paired_comparisons, pareto_flags
        from safebench.policies import fit_model
        from safebench.report import generate_report
        scenarios = tuple(Scenario(sid, tuple(MarketObservation(sid, year, (0., 0., 0.), 0.)
            for year in range(1, 31))) for sid in (1, 2))
        config = dict(CONFIG, selection_rule='one_se_utility')
        model = fit_model(scenarios[:1], config)
        metrics, records = evaluate(scenarios[1:], model, {}, config, names=['equal_annual', 'balanced_annual'])
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark') as td:
            root = Path(td)
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            (root / 'manifest.json').write_text(json.dumps(dict(phase='development', scenario_ids=[2])), encoding='utf-8')
            metrics.to_csv(root / 'scenario_metrics.csv', index=False)
            records.to_csv(root / 'decisions.csv.gz', index=False)
            pareto_flags(summarize(metrics)).to_csv(root / 'summary.csv', index=False)
            paired_comparisons(metrics, bootstrap=50).to_csv(root / 'paired_comparisons.csv', index=False)
            report = generate_report(root)
            self.assertIn('开发集交叉验证', report.read_text(encoding='utf-8'))
            self.assertIn('策略', (root / 'comparison_table.csv').read_text(encoding='utf-8'))
            self.assertEqual(len(list((root / 'figures').glob('*.pdf'))), 5)
            self.assertEqual(len(list((root / 'figures').glob('*.png'))), 5)

    def write_sources(self, root):
        for label, rule in (('control', 'one_se_utility'), ('constrained', 'path_risk')):
            path = root / label
            path.mkdir()
            config = dict(CONFIG, selection_rule=rule, bootstrap=100)
            saved_manifest = dict(phase='development', holdout_used=False, scenario_ids=list(range(1, 11)),
                code_sha256='same-code', partition_sha256='same-partition', dependencies={}, python='fixture',
                config_sha256=digest(config))
            (path / 'config.json').write_text(json.dumps(config))
            (path / 'manifest.json').write_text(json.dumps(saved_manifest))
            (path / 'splits.json').write_text('[]')
            rows = [dict(policy=name, scenario_id=sid, horizon=30, cost_rate=.0025,
                fold=sid % 2, real_cagr=sid * .001, real_max_drawdown=sid * .01,
                trade_rate=.1, total_real_cost=.2) for name in ('cost_aware', 'adaptive') for sid in range(1, 11)]
            pd.DataFrame(rows).to_csv(path / 'scenario_metrics.csv', index=False)
        return root / 'control', root / 'constrained'

    def test_identical_pair_has_zero_intervals_and_family_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            control, constrained = self.write_sources(root)
            result = compare_selection_runs(control, constrained, root / 'comparison')
            self.assertTrue((result[['difference','lower95','upper95','family_lower','family_upper']] == 0).all().all())
            self.assertTrue((result.comparison_count == 2).all())

    def test_changed_search_budget_is_not_a_selection_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            control, constrained = self.write_sources(root)
            config = json.loads((control / 'config.json').read_text())
            config['tuning_grid'] = config['tuning_grid'][:2]
            (control / 'config.json').write_text(json.dumps(config))
            saved = json.loads((control / 'manifest.json').read_text())
            saved['config_sha256'] = digest(config)
            (control / 'manifest.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'search budget'):
                compare_selection_runs(control, constrained, root / 'comparison')

    def test_final_results_are_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            control, constrained = self.write_sources(root)
            saved = json.loads((constrained / 'manifest.json').read_text())
            saved['phase'] = 'final'
            (constrained / 'manifest.json').write_text(json.dumps(saved))
            with self.assertRaises(PermissionError):
                compare_selection_runs(control, constrained, root / 'comparison')


if __name__ == '__main__':
    unittest.main()
