import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from safebench.incremental import CANDIDATE, REFERENCES, incremental_analysis


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(seed=7,cost_rate=.0025,bootstrap=200,budget_value_screen=dict(
            confidence=.95,minimum_p90_reduction=.01,maximum_median_growth_loss=.001,
            maximum_p10_growth_loss=.001,minimum_consistent_folds=4,
            maximum_breach_upper=.20,maximum_review_rate=.20))

    def frame(self):
        rows=[]
        for horizon,cost in [(5,.0025),(30,.0025),(30,.005)]:
            for name in [CANDIDATE,CANDIDATE+'_zero_band',*REFERENCES]:
                for sid in range(1,41):
                    dynamic=name==CANDIDATE
                    rows.append(dict(policy=name,horizon=horizon,cost_rate=cost,
                        scenario_id=sid,fold=(sid-1)%5,real_cagr=.04+sid*.00001,
                        real_max_drawdown=(.10 if dynamic else .15)+sid*.00001,
                        review=0.,trade_rate=.2 if dynamic else .4,
                        total_real_cost=.5 if dynamic else 1.))
        return pd.DataFrame(rows)

    def test_adjusted_screen_and_band_are_paired(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            self.frame().to_csv(out/'scenario_metrics.csv',index=False)
            incremental_analysis(out,self.config)
            comparisons=pd.read_csv(out/'incremental_comparisons.csv')
            screen=pd.read_csv(out/'incremental_screen.csv')
            self.assertEqual(set(comparisons.family_size),{6})
            self.assertEqual(len(comparisons),18)
            self.assertTrue(screen.incremental_value_established.all())
            self.assertTrue((screen.risk_consistent_folds==5).all())
            bands=pd.read_csv(out/'band_ablation.csv')
            self.assertTrue((bands.trade_rate_difference<0).all())
            self.assertFalse(json.loads((out/'incremental_protocol.json').read_text())['holdout_used'])

    def test_point_growth_loss_cannot_be_hidden_by_risk_improvement(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); frame=self.frame()
            frame.loc[frame.policy==CANDIDATE,'real_cagr']-=.005
            frame.to_csv(out/'scenario_metrics.csv',index=False)
            incremental_analysis(out,self.config)
            screen=pd.read_csv(out/'incremental_screen.csv')
            self.assertFalse(screen.incremental_value_established.any())
            self.assertTrue(screen.risk_reduction_pass.all())

    def test_missing_paired_scenario_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            self.frame().drop(index=80).to_csv(out/'scenario_metrics.csv',index=False)
            with self.assertRaisesRegex(ValueError,'paired'):
                incremental_analysis(out,self.config)

    def test_missing_double_cost_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); frame=self.frame()
            frame[frame.cost_rate==.0025].to_csv(out/'scenario_metrics.csv',index=False)
            with self.assertRaisesRegex(ValueError,'double-cost'):
                incremental_analysis(out,self.config)
