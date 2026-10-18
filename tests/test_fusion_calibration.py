"""Generated checks for constrained grouped selection and actual execution scores."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_calibration import choose_group, inner_scores
from safeinvest.contracts import SimulationConfig
from safebench.engine import simulate
from safebench.configuration import load_config
from safebench.metrics import compute_metrics
from safebench.plan_a import PlanAPolicy, parameter_grid
from test_plan_bc import fixture


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'configs/submission.json')
        self.horizons = [15,25,30]
        self.ids = [90,91,92,93]
        self.model = {'plan_a': {'entries': {f'balanced:{h}:{ratio:g}': {'status':'optimal'}
                      for h in self.horizons for ratio in self.config['plan_a']['risk_ratios']}}}
        self.rows = pd.DataFrame([dict(family='fusion', trial=t, scenario_id=sid, horizon=h,
            real_cagr=.04+t*.0002, annual_growth=.04, real_max_drawdown=.1,
            plan_a_validation_budget=.5, plan_a_risk_budget=.4, review=0.)
            for t in range(6) for sid in self.ids for h in self.horizons])

    def choose(self, rows=None, unavailable=None):
        return choose_group(self.rows if rows is None else rows, self.model, self.config,
                            self.horizons, self.ids, 'fusion', unavailable or {})

    def test_feasibility_beats_growth_and_preserves_global_horizons(self):
        self.rows.loc[self.rows.trial==5, 'real_max_drawdown'] = .8
        old = copy.deepcopy(self.config)
        chosen, trials = self.choose()
        self.assertEqual(chosen['trial'], 4)
        self.assertTrue(chosen['selection_feasible'])
        self.assertFalse(trials[5]['selection_feasible'])
        self.assertEqual(self.config, old)

    def test_no_feasible_candidate_remains_explicit_and_ties_are_stable(self):
        self.rows.real_max_drawdown = .8+self.rows.trial*.01
        chosen, trials = self.choose()
        self.assertEqual(chosen['trial'], 0)
        self.assertFalse(chosen['selection_feasible'])
        self.assertEqual(len(trials), 6)
        self.assertGreater(chosen['constraint_violation'], 0)
        self.rows.real_max_drawdown = .1
        self.rows.real_cagr = .04
        self.assertEqual(self.choose()[0]['trial'], 0)

    def test_incomplete_duplicate_and_allocation_failure(self):
        with self.assertRaises(ValueError): self.choose(self.rows.iloc[1:])
        with self.assertRaises(ValueError): self.choose(pd.concat([self.rows, self.rows.iloc[:1]]))
        self.rows.loc[self.rows.trial==5, 'real_max_drawdown'] = .8
        chosen, trials = self.choose(unavailable={4:[1]})
        self.assertEqual(chosen['trial'], 3)
        self.assertEqual(trials[4]['inner_infeasible_folds'], [1])

    def test_inner_scores_are_actual_review_paths_and_reject_overlap(self):
        scenarios, config, model, params = fixture()
        base = json.loads((ROOT/'configs/fusion_research.json').read_text())
        rule = dict(buffer=.02, short_a_max_horizon=6)
        s = replace(scenarios[0], scenario_id=100,
                    observations=tuple(replace(o,scenario_id=100) for o in scenarios[0].observations))
        for h in [5,30]: model['plan_a']['entries'][f'balanced:{h}:1']['review_threshold'] = 0.
        scores = inner_scores([s], model, params, config, [5,30], base, rule)
        for row in scores:
            self.assertEqual(row['review'], 1.)
            if row['family'] == 'a':
                policy = PlanAPolicy('plan_a_balanced',model,params,config,row['horizon'])
                expected = compute_metrics(simulate(s,policy,policy.initial_target,
                    SimulationConfig(row['horizon'],config['initial_wealth'],config['cost_rate'])))
                self.assertEqual(row['real_cagr'], expected['real_cagr'])
                self.assertEqual(row['real_max_drawdown'], expected['real_max_drawdown'])
        with self.assertRaises(ValueError): inner_scores([scenarios[0]],model,params,config,[5],base,rule)
