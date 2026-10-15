"""Independent audit regression tests using generated market paths only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from safeinvest.contracts import Action, Decision, MarketObservation, Scenario, SimulationConfig
from safebench.engine import simulate
from safebench.metrics import compute_metrics
from safebench.policies import StaticPolicy
from safebench.validation import audit_outputs


class FixtureGate:
    name = 'budget_lifecycle_balanced'
    requires_benefit_gate = True
    buffer = .002
    initial_target = (.6, .3, .1)

    def decide(self, context):
        target = (.3, .5, .2) if context.period % 2 == 0 else self.initial_target
        current = [x / sum(context.holdings) for x in context.holdings]
        cost = context.transaction_cost_rate * sum(abs(a-b) for a, b in zip(target, current)) / 2
        return Decision(Action.CHANGE_TARGET, ('FIXTURE_GATE',), 'Generated gated trade.',
                        target, .05, cost)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = dict(initial_wealth=100.)
        self.metrics = []
        self.records = []
        (self.out/'manifest.json').write_text(json.dumps(dict(phase='development', scenario_ids=[1])))
        self.scenario = Scenario(1, tuple(
            MarketObservation(1, period, ((.12, -.18, .04)[(period-1) % 3], .025, .01), .015)
            for period in range(1, 31)))

    def add_path(self, policy, horizons=(5, 6, 15, 25, 30), cost=.0025, explicit=False, fold=0):
        cfg = SimulationConfig(max(horizons), 100., cost)
        result = simulate(self.scenario, policy, policy.initial_target, cfg)
        for horizon in horizons:
            prefix = replace(result, config=replace(cfg, horizon=horizon), records=result.records[:horizon])
            self.metrics.append(dict(compute_metrics(prefix), fold=fold))
        for record in result.records:
            row = asdict(record)
            for field in ('target_weights', 'pre_trade_holdings', 'post_trade_holdings', 'end_holdings'):
                row.update({field+'_'+asset: value for asset, value in
                            zip(('equity', 'bond', 'bill'), row.pop(field))})
            row.update(policy=policy.name, scenario_id=1, horizon=max(horizons), cost_rate=cost, fold=fold)
            row['action'] = record.action.value
            row['reason_codes'] = json.dumps(row['reason_codes'])
            if explicit:
                row['requires_benefit_gate'] = policy.requires_benefit_gate
                row['buffer'] = getattr(policy, 'buffer', 0.)
            self.records.append(row)

    def save(self):
        (self.out/'config.json').write_text(json.dumps(self.config))
        pd.DataFrame(self.metrics).to_csv(self.out/'scenario_metrics.csv', index=False)
        pd.DataFrame(self.records).to_csv(self.out/'decisions.csv.gz', index=False, compression='gzip')

    def audit(self):
        self.save()
        return audit_outputs(self.out)

    def test_full_path_audits_every_prefix_and_cost(self):
        for cost in (0., .0025, .005):
            self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)), cost=cost)
        result = self.audit()
        self.assertEqual(result['audited_paths'], 3)
        self.assertEqual(result['audited_metric_rows'], 15)
        self.assertEqual(result['metric_rows'], 15)
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(result['gate_cost_checked_trades'], 0)

    def test_each_short_term_metric_tampering_is_detected(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        fields = [key for key in self.metrics[0] if key not in
                  ('scenario_id', 'policy', 'horizon', 'cost_rate', 'fold')]
        for horizon in (5, 6, 15, 25):
            for field in fields:
                with self.subTest(horizon=horizon, field=field):
                    metric = next(row for row in self.metrics if row['horizon'] == horizon)
                    original = metric[field]
                    metric[field] += .1
                    with self.assertRaises(AssertionError):
                        self.audit()
                    metric[field] = original

    def test_nonfinite_short_term_metrics_fail(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        for field in ('real_cagr', 'real_max_drawdown', 'terminal_real_wealth'):
            with self.subTest(field=field):
                original = self.metrics[0][field]
                self.metrics[0][field] = float('nan')
                with self.assertRaises(AssertionError):
                    self.audit()
                self.metrics[0][field] = original

    def test_independent_horizons_do_not_reuse_long_path(self):
        name = 'budget_lifecycle_balanced_static'
        self.add_path(StaticPolicy(name, (.2, .5, .3)), horizons=(5,))
        self.add_path(StaticPolicy(name, (.6, .3, .1)), horizons=(30,))
        result = self.audit()
        self.assertEqual(result['audited_paths'], 2)
        self.assertEqual(result['audited_metric_rows'], 2)
        self.metrics[0]['real_cagr'] += .01
        with self.assertRaises(AssertionError):
            self.audit()

    def test_legacy_lifecycle_missing_short_path_fails(self):
        name = 'drawdown_lifecycle_balanced'
        self.add_path(StaticPolicy(name, (.2, .5, .3)), horizons=(5,))
        self.add_path(StaticPolicy(name, (.6, .3, .1)), horizons=(30,))
        self.records = [row for row in self.records if row['horizon'] == 30]
        with self.assertRaisesRegex(AssertionError, 'Missing metric path'):
            self.audit()

    def test_new_static_comparator_missing_short_path_fails(self):
        self.add_path(StaticPolicy('budget_lifecycle_balanced_static', (.2, .5, .3)), horizons=(5, 30))
        with self.assertRaisesRegex(AssertionError, 'Missing metric path'):
            self.audit()

    def test_generic_independent_horizons_are_matched_once(self):
        for horizon, weights in ((5, (.2, .5, .3)), (15, (.4, .4, .2)), (30, (.6, .3, .1))):
            self.add_path(StaticPolicy('future_dynamic', weights), horizons=(horizon,))
        self.assertEqual(self.audit()['audited_metric_rows'], 3)
        self.records = [row for row in self.records if row['horizon'] != 15]
        with self.assertRaisesRegex(AssertionError, 'Missing metric path'):
            self.audit()

    def test_missing_policy_records_cannot_leave_unchecked_metrics(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        self.add_path(StaticPolicy('scheme_a_dynamic', (.3, .5, .2)))
        self.records = [row for row in self.records if row['policy'] == 'scheme_a_static']
        with self.assertRaisesRegex(AssertionError, 'Missing metric path'):
            self.audit()

    def test_missing_terminal_or_middle_record_fails(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        original = self.records[:]
        for period in (1, 3, 30):
            with self.subTest(period=period):
                self.records = [row for row in original if row['period'] != period]
                with self.assertRaises(AssertionError):
                    self.audit()

    def test_metric_beyond_saved_path_fails(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)), horizons=(5,))
        self.metrics.append(dict(self.metrics[0], horizon=6))
        with self.assertRaisesRegex(AssertionError, 'Missing metric path'):
            self.audit()

    def test_missing_configured_metric_horizon_fails(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        self.config.update(horizons=[5, 6, 15, 25, 30], cost_rate=.0025)
        self.metrics = [row for row in self.metrics if row['horizon'] != 5]
        with self.assertRaisesRegex(AssertionError, 'Missing configured metric horizons'):
            self.audit()

    def test_missing_terminal_metric_fails(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        self.metrics = self.metrics[:-1]
        with self.assertRaisesRegex(AssertionError, 'Missing terminal metric'):
            self.audit()

    def test_duplicate_metric_or_record_fails(self):
        self.add_path(StaticPolicy('scheme_a_static', (.5, .35, .15)))
        self.metrics.append(dict(self.metrics[0]))
        with self.assertRaises(AssertionError):
            self.audit()
        self.metrics.pop()
        self.records.append(dict(self.records[0]))
        with self.assertRaises(AssertionError):
            self.audit()

    def test_explicit_new_gate_fields_and_initial_allocation(self):
        self.add_path(FixtureGate(), horizons=(5,), explicit=True)
        result = self.audit()
        self.assertEqual(result['gate_cost_checked_trades'], 4)
        self.assertEqual(result['gate_full_buffer_checked_trades'], 4)
        self.assertEqual(result['gate_missing_buffer_trades'], 0)
        self.assertEqual(result['gate_exempt_trades'], 0)

    def test_gate_requires_strict_benefit_above_cost_plus_buffer(self):
        self.add_path(FixtureGate(), horizons=(5,), explicit=True)
        trade = self.records[1]
        for delta in (-.001, 0.):
            with self.subTest(delta=delta):
                trade['estimated_benefit'] = trade['estimated_cost'] + trade['buffer'] + delta
                with self.assertRaisesRegex(AssertionError, 'benefit/cost gate'):
                    self.audit()

    def test_gate_cost_estimate_tampering_fails(self):
        self.add_path(FixtureGate(), horizons=(5,), explicit=True)
        self.records[1]['estimated_cost'] += .001
        with self.assertRaisesRegex(AssertionError, 'cost estimate mismatch'):
            self.audit()

    def test_explicit_gate_missing_estimates_fail(self):
        self.add_path(FixtureGate(), horizons=(5,), explicit=True)
        for field in ('estimated_cost', 'estimated_benefit'):
            with self.subTest(field=field):
                original = self.records[1][field]
                self.records[1][field] = None
                with self.assertRaises(AssertionError):
                    self.audit()
                self.records[1][field] = original

    def test_saved_fold_parameters_supply_legacy_buffer(self):
        self.add_path(FixtureGate(), horizons=(5,), fold=3)
        (self.out/'fitted_models.json').write_text(json.dumps([
            dict(fold=0, params={FixtureGate.name: dict(buffer=0.)}),
            dict(fold=3, params={FixtureGate.name: dict(buffer=.002)})]))
        self.assertEqual(self.audit()['gate_full_buffer_checked_trades'], 4)
        self.records[1]['estimated_benefit'] = self.records[1]['estimated_cost'] + .001
        with self.assertRaisesRegex(AssertionError, 'benefit/cost gate'):
            self.audit()

    def test_frozen_parameters_supply_buffer(self):
        self.add_path(FixtureGate(), horizons=(5,))
        (self.out/'frozen_bundle.json').write_text(json.dumps(dict(
            params={FixtureGate.name: dict(buffer=.002)})))
        self.assertEqual(self.audit()['gate_full_buffer_checked_trades'], 4)

    def test_explicit_buffer_wins_over_saved_parameters(self):
        self.add_path(FixtureGate(), horizons=(5,), explicit=True)
        (self.out/'fitted_models.json').write_text(json.dumps([
            dict(fold=0, params={FixtureGate.name: dict(buffer=.5)})]))
        self.assertTrue(self.audit()['all_checks_passed'])

    def test_missing_buffer_is_reported_not_claimed_fully_verified(self):
        self.add_path(FixtureGate(), horizons=(5,))
        result = self.audit()
        self.assertEqual(result['gate_missing_buffer_trades'], 4)
        self.assertEqual(result['gate_full_buffer_checked_trades'], 0)
        self.assertIn('not verified', result['gate_check_limitations'])
        self.records[1]['estimated_benefit'] = self.records[1]['estimated_cost']
        with self.assertRaisesRegex(AssertionError, 'benefit/cost gate'):
            self.audit()

    def test_diagnostic_comparator_is_not_subject_to_gate(self):
        self.add_path(StaticPolicy('balanced_annual', (.5, .35, .15)), horizons=(5,), explicit=True)
        result = self.audit()
        self.assertEqual(result['gate_cost_checked_trades'], 0)
        self.assertEqual(result['gate_unclassified_or_diagnostic_trades'], 4)

    def test_legacy_exemption_keeps_direction_and_cost_checks(self):
        self.add_path(FixtureGate(), horizons=(2,), explicit=True)
        self.records[1]['gate_exempt'] = True
        self.records[1]['estimated_benefit'] = -1.
        result = self.audit()
        self.assertEqual(result['gate_exempt_checked_trades'], 1)
        self.records[1]['estimated_cost'] += .01
        with self.assertRaisesRegex(AssertionError, 'cost estimate mismatch'):
            self.audit()

    def test_exemption_cannot_bypass_equity_direction(self):
        self.add_path(FixtureGate(), horizons=(3,), explicit=True)
        self.records[2]['gate_exempt'] = True
        with self.assertRaisesRegex(AssertionError, 'do not reduce equity'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
