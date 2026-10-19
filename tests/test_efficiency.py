"""Generated fixtures for training-only utility gates and the efficiency protocol."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from safebench import ROOT
from safeinvest.contracts import Scenario, MarketObservation, PolicyContext, SimulationConfig, Action
from safebench.scenario_utility import fit_utility_blocks, log_certainty_equivalent, ScenarioUtilityPolicy, CE_POLICY_NAMES
from safebench.policies import PORTFOLIOS, make_policy, fit_model
from safebench.engine import simulate
from safebench.efficiency import select_efficiency_trials, efficiency_intervals, candidate_suitability
from safebench.experiment import tune, evaluate, freeze, manifest

CONFIG = json.loads((ROOT / 'configs/efficiency.json').read_text(encoding='utf-8'))
ANCHOR = PORTFOLIOS['balanced']


def scenario(sid=1, rows=None):
    rows = rows or [(0., 0., 0., 0.)] * 30
    return Scenario(sid, tuple(MarketObservation(sid, year, tuple(row[:3]), row[3])
        for year, row in enumerate(rows, 1)))


def context(holdings=(70., 20., 10.), cost=.0025):
    return PolicyContext(2, (), holdings, ANCHOR, 1., 100., False, cost)


def defensive_blocks():
    return dict(gross=[[.5, 1., 1.], [1.1, 1., 1.]], probabilities=[.5, .5])


class BlockValuationTests(unittest.TestCase):
    def test_contiguous_compounding_inflation_and_equal_scenario_mass(self):
        a = scenario(1, [( .1, .02, .01, .05)] * 3)
        b = scenario(2, [(0., 0., 0., 0.)] * 4)
        blocks = fit_utility_blocks((a, b), (1, 3))
        block = blocks['3']
        np.testing.assert_allclose(block['gross'][0], (np.array([1.1, 1.02, 1.01]) / 1.05) ** 3)
        self.assertEqual(block['scenario_ids'], [1, 2, 2])
        self.assertEqual(block['start_periods'], [1, 1, 2])
        self.assertEqual(block['probabilities'], [.5, .25, .25])

    def test_no_final_or_duplicate_training_ids(self):
        for scenarios in ((scenario(121),), (scenario(), scenario())):
            with self.assertRaises(PermissionError):
                fit_utility_blocks(scenarios)

    def test_bad_training_returns_fail(self):
        with self.assertRaises(ValueError):
            fit_utility_blocks((scenario(rows=[(-1., 0., 0., 0.)] * 3),))

    def test_crra_hand_value_and_log_case(self):
        factors = np.array([[.5, .5, .5], [2., 2., 2.]])
        observed = np.exp(log_certainty_equivalent(factors, ANCHOR, 5., [.5, .5]))
        self.assertAlmostEqual(observed, (.5 * .5 ** -4 + .5 * 2 ** -4) ** (-.25))
        self.assertAlmostEqual(np.exp(log_certainty_equivalent(factors, ANCHOR, 1., [.5, .5])), 1.)

    def test_ce_homogeneity_for_upfront_fee(self):
        block = defensive_blocks()
        gross = np.array(block['gross'])
        a = np.exp(log_certainty_equivalent(gross, ANCHOR, 5., block['probabilities']))
        b = np.exp(log_certainty_equivalent(gross * .99, ANCHOR, 5., block['probabilities']))
        self.assertAlmostEqual(b, .99 * a)


class UtilityPolicyTests(unittest.TestCase):
    def policy(self, buffer=0.):
        return ScenarioUtilityPolicy('utility_band_1y', defensive_blocks(), dict(band=.03, buffer=buffer), CONFIG, ANCHOR)

    def test_band_hold_and_cost_justified_rebalance(self):
        policy = self.policy()
        self.assertEqual(policy.decide(context((50., 35., 15.))).reason_codes, ('WITHIN_TRADE_BAND',))
        decision = policy.decide(context())
        self.assertEqual(decision.action, Action.REBALANCE)
        self.assertEqual(decision.target_weights, ANCHOR)
        self.assertAlmostEqual(decision.estimated_cost, .0025 * .2)
        self.assertGreater(decision.estimated_benefit, decision.estimated_cost)
        self.assertLessEqual(len(decision.explanation.split()), 50)

    def test_large_cost_prevents_trade(self):
        self.assertEqual(self.policy().decide(context(cost=.99)).action, Action.KEEP)

    def test_review_precedes_band_and_is_absorbing(self):
        policy = self.policy()
        c = context((30., 21., 9.))
        self.assertEqual(policy.decide(c).action, Action.HUMAN_REVIEW)
        path = scenario(10, [(-.6, -.6, -.6, 0.)] + [(0., 0., 0., 0.)] * 3)
        result = simulate(path, policy, ANCHOR, SimulationConfig(4))
        self.assertEqual(result.records[1].action, Action.HUMAN_REVIEW)
        self.assertEqual(result.records[2].reason_codes, ('REVIEW_PENDING',))
        self.assertTrue(all(r.transaction_cost == 0 for r in result.records[1:]))

    def test_nonfinite_runtime_estimate_requests_review(self):
        policy = self.policy()
        policy.gross[0, 0] = float('nan')
        self.assertEqual(policy.decide(context()).reason_codes, ('INVALID_ESTIMATE',))

    def test_future_perturbation_does_not_change_prior_decisions(self):
        model = dict(ce_blocks={'1': defensive_blocks()})
        params = dict(band=.03, buffer=0.)
        base = scenario(10, [( .1, 0., 0., .01)] * 5)
        changed = replace(base, observations=base.observations[:4] + (MarketObservation(10, 5, (-.9, 2., 1.), .2),))
        a = simulate(base, make_policy('utility_band_1y', model, params, CONFIG), ANCHOR, SimulationConfig(5))
        b = simulate(changed, make_policy('utility_band_1y', model, params, CONFIG), ANCHOR, SimulationConfig(5))
        self.assertEqual(a.records[:4], b.records[:4])
        self.assertEqual(a.records[4].action, b.records[4].action)
        self.assertEqual(a.records[4].post_trade_holdings, b.records[4].post_trade_holdings)


class EfficiencySelectionTests(unittest.TestCase):
    def data(self):
        ref = pd.DataFrame([dict(policy='balanced_annual', scenario_id=sid, real_cagr=.04,
            real_max_drawdown=.3, trade_rate=1., total_real_cost=1.) for sid in range(1, 11)])
        rows = [dict(policy=name, scenario_id=sid, trial=trial, real_cagr=.04 if trial else .02,
            real_max_drawdown=.3, trade_rate=.2, total_real_cost=.5) for name in CE_POLICY_NAMES
            for trial in range(2) for sid in range(1, 11)]
        return pd.DataFrame(rows), ref, dict(CONFIG, ce_tuning_grid=CONFIG['ce_tuning_grid'][:2])

    def test_selection_rejects_low_growth_even_with_few_trades(self):
        scores, ref, config = self.data()
        chosen, rows = select_efficiency_trials(scores, ref, config)
        self.assertTrue(all(p['selection_trial'] == 1 for p in chosen.values()))
        self.assertTrue(all(r['risk_feasible'] for r in rows if r['selected']))

    def test_infeasibility_is_explicit_and_unpaired_data_rejected(self):
        scores, ref, config = self.data()
        scores.real_cagr = .01
        chosen, _ = select_efficiency_trials(scores, ref, config)
        self.assertTrue(all(p['selection_status'] == 'infeasible_diagnostic' for p in chosen.values()))
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            select_efficiency_trials(scores.iloc[:-1], ref, config)

    def test_nested_trace_and_model_provenance(self):
        scenarios = tuple(scenario(sid) for sid in range(1, 13))
        config = dict(CONFIG, inner_folds=2, ce_tuning_grid=CONFIG['ce_tuning_grid'][:2])
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark') as td:
            path = Path(td) / 'tuning_metrics/outer_0.csv.gz'
            _, trials, audits = tune(scenarios, config, 42, path)
            trace = pd.read_csv(path)
            self.assertEqual(len(trace), 12 * 5)
            self.assertEqual(len(trials), 4)
            for inner, split in enumerate(audits):
                model = json.loads((Path(td) / f'tuning_models/outer_0_inner_{inner}.json').read_text())
                self.assertTrue(set(model['train_ids']).isdisjoint(split['validation_ids']))
                self.assertTrue(set(model['ce_blocks']['3']['scenario_ids']) <= set(model['train_ids']))
            with self.assertRaises(FileExistsError):
                tune(scenarios, config, 42, path)

    def test_paired_intervals_cover_cost_strata_and_trade_ratio(self):
        scores, ref, _ = self.data()
        rows = []
        for cost in (.0025, .005):
            for name in ('balanced_annual', *CE_POLICY_NAMES):
                for sid in range(1, 11):
                    rows.append(dict(policy=name, scenario_id=sid, horizon=30, cost_rate=cost,
                        real_cagr=.04, real_max_drawdown=.3, trade_rate=1. if name == 'balanced_annual' else .2,
                        total_real_cost=1. if name == 'balanced_annual' else .5))
        result = efficiency_intervals(pd.DataFrame(rows), dict(CONFIG, efficiency_bootstrap=100))
        self.assertEqual(len(result), 20)
        self.assertTrue(result.passed.all())
        self.assertTrue((result.family_comparisons == 20).all())
        self.assertTrue(np.allclose(result[result.metric == 'trade_constraint'].difference, -.3))

    def test_customer_risk_uses_stated_horizon_and_five_choice_bound(self):
        rows = [dict(policy=name, scenario_id=sid, horizon=horizon, cost_rate=.0025,
            real_cagr=.04, real_max_drawdown=.05 if horizon == 5 else .4)
            for name in CE_POLICY_NAMES for sid in range(1, 121) for horizon in (5, 15)]
        profiles = [dict(profile_id='A', horizon=5, loss_tolerance=.1, liquidity='high'),
                    dict(profile_id='B', horizon=15, loss_tolerance=.25, liquidity='medium')]
        result = candidate_suitability(pd.DataFrame(rows), profiles, CONFIG)
        self.assertTrue((result[result.profile == 'A'].breaches == 0).all())
        self.assertTrue((result[result.profile == 'B'].breaches == 120).all())
        self.assertTrue(np.allclose(result.confidence, .99))
        self.assertFalse(result.eligible.any())

    def test_infeasible_efficiency_candidate_freeze_is_rejected_before_market_read(self):
        config = dict(CONFIG, outer_folds=1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'manifest.json').write_text(json.dumps(manifest(config, 'fixture', [1], 'development')))
            pd.DataFrame([dict(policy='utility_band_1y', selected=True, risk_feasible=False)]).to_csv(root / 'tuning_trials.csv', index=False)
            pd.DataFrame([dict(policy='utility_band_1y', all_checks_passed=False)]).to_csv(root / 'efficiency_screen.csv', index=False)
            with patch('safebench.experiment.load_partition', side_effect=AssertionError('Unexpected market read')):
                with self.assertRaisesRegex(ValueError, 'efficiency diagnostic'):
                    freeze(config, root / 'unused.csv', 'utility_band_1y', root / 'frozen.json', root)

    def test_development_pipeline_and_reports_on_generated_market_only(self):
        from safebench.experiment import run_development, code_digest
        fixtures = tuple(scenario(sid) for sid in range(1, 13))
        config = dict(CONFIG, outer_folds=3, inner_folds=2, ce_tuning_grid=CONFIG['ce_tuning_grid'][:1],
            bootstrap=50, efficiency_bootstrap=100, sensitivity_costs=[.005])
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark') as td:
            out = Path(td) / 'generated_run'
            with patch('safebench.experiment.load_partition', return_value=(fixtures, 'generated-fixture')):
                run_development(config, Path(td) / 'never_read_market.csv', out)
            self.assertTrue(json.loads((out / 'validation.json').read_text())['all_checks_passed'])
            self.assertEqual(json.loads((out / 'manifest.json').read_text())['code_sha256'], code_digest())
            metrics = pd.read_csv(out / 'scenario_metrics.csv')
            self.assertEqual(set(metrics.scenario_id), set(range(1, 13)))
            self.assertEqual(len(pd.read_csv(out / 'efficiency_intervals.csv')), 20)
            self.assertEqual(len(pd.read_csv(out / 'candidate_suitability.csv')), 10)
            self.assertEqual(len(list((out / 'tuning_models').glob('*.json'))), 6)
            self.assertEqual(len(list((out / 'figures').glob('*.pdf'))), 8)
            self.assertIn('交易效率研究', (out / 'efficiency_report.html').read_text(encoding='utf-8'))
            self.assertIn('一年情景效用门控', (out / 'comparison_table.csv').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
