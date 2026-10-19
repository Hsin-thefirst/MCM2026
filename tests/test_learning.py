"""Generated-market tests for causal counterfactual fitting and horizon-specific deployment."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from safebench import ROOT
from safeinvest.contracts import Scenario, MarketObservation, SimulationConfig, Action
from safebench.engine import simulate, execute_trade
from safebench.learning_core import (counterfactual_data, fit_fqi, fit_tree, predict_tree,
    encode_values, decode_values, choose_actions, FQIPolicy, FEATURE_NAMES, LEARNING_NAMES)
from safebench.learning_study import evaluate_learning, select_learning_trials, learning_intervals
from safebench.experiment import run_development, freeze

CONFIG=json.loads((ROOT/'configs/learning.json').read_text())
ANCHOR=(.5,.35,.15)


def scenario(sid=1, rows=None):
    rows=rows or [(0.,0.,0.,0.)]*30
    return Scenario(sid,tuple(MarketObservation(sid,t,tuple(row[:3]),row[3]) for t,row in enumerate(rows,1)))


def manual_model():
    models={}
    for k in range(1,30):
        values=[0.,.2,0.,0.] if k%2==0 else [0.,0.,0.,.2]
        tree=dict(value=values,n=100)
        models[str(k)]=dict(teacher=tree,student=tree)
    fit=dict(maximum_remaining=29,models=models)
    return dict(train_ids=[1],fqi={'full':{'0.25':fit},'portfolio':{'0.25':fit}})


class CounterfactualTests(unittest.TestCase):
    def test_rejects_holdout_duplicate_and_invalid_returns(self):
        for paths in ((scenario(121),),(scenario(),scenario())):
            with self.assertRaises(PermissionError):counterfactual_data(paths,CONFIG)
        with self.assertRaises(ValueError):
            counterfactual_data((scenario(rows=[(-1.,0.,0.,0.)]*3),),CONFIG)

    def test_transitions_match_engine_trade_and_purchasing_power(self):
        path=scenario(rows=[(.12,.02,.01,.04),(-.20,.03,.02,.05),(.1,0.,0.,.02)])
        data=counterfactual_data((path,),CONFIG)
        self.assertEqual(data['x'].shape,(16,len(FEATURE_NAMES)))
        self.assertEqual(data['next_x'].shape,(16,4,len(FEATURE_NAMES)))
        row=0;x=data['x'][row]
        holdings=np.array(execute_trade((0.,0.,100.),ANCHOR,.0025).post_trade_holdings)
        holdings*=1+np.array(path.observations[0].nominal_returns)
        high=max(100.,holdings.sum()/1.04)
        for action in range(4):
            trade=execute_trade(holdings,CONFIG['learning_targets'][action-1],.0025) if action else None
            post=np.array(trade.post_trade_holdings) if action else holdings
            end=post*(1+np.array(path.observations[1].nominal_returns))
            real=end.sum()/(1.04*1.05)
            self.assertAlmostEqual(data['fees'][row,action],trade.transaction_cost/holdings.sum() if action else 0.)
            self.assertAlmostEqual(data['gross_log'][row,action],np.log(end.sum()/post.sum()/1.05))
            self.assertAlmostEqual(data['next_x'][row,action,2],1-real/max(high,real))
            np.testing.assert_allclose(data['next_x'][row,action,:2],(end/end.sum())[:2])
        np.testing.assert_allclose(x[4:7],np.array([1.12,1.02,1.01])/1.04-1)

    def test_breach_reward_occurs_once_and_review_forces_hold(self):
        path=scenario(rows=[(0.,0.,0.,0.),(-.9,-.9,0.,0.),(-.1,-.1,0.,0.),(0.,0.,0.,0.)])
        data=counterfactual_data((path,),CONFIG)
        rows=np.flatnonzero((data['behavior']==2)&np.isclose(data['x'][:,10],.0025))
        self.assertTrue(data['breach'][rows[0],2])
        self.assertEqual(float(data['breach'][rows[1:]].sum()),0.)
        self.assertTrue(np.allclose(data['fees'][rows[1:]],0.))
        np.testing.assert_allclose(data['next_x'][rows[1],0],data['next_x'][rows[1],3])

    def test_tree_fit_and_q_encoding(self):
        x=np.arange(12,dtype=float)[:,None]
        y=np.where(x<6,1.,3.)
        tree=fit_tree(x,y,2,2,16,[0])
        np.testing.assert_allclose(predict_tree(tree,x),y)
        q=np.array([[.1,.2,-.1,.4]])
        np.testing.assert_allclose(decode_values(encode_values(q,10.),10.),q)

    def test_adjacent_floats_preserve_training_leaf_membership(self):
        a=1.;b=np.nextafter(a,np.inf)
        x=np.array([[a],[a],[b],[b]])
        y=np.array([[0.],[0.],[1.],[1.]])
        tree=fit_tree(x,y,1,2,16,[0])
        np.testing.assert_array_equal(predict_tree(tree,x),y)
        self.assertEqual(tree['threshold'],a)

    def test_finite_horizon_bellman_and_distillation_trace(self):
        data=counterfactual_data((scenario(rows=[(.1,.02,0.,0.)]*5),),CONFIG)
        config=dict(CONFIG,learning_min_leaf=2,learning_teacher_depth=2)
        fitted=fit_fqi(data,.25,'portfolio',config)
        self.assertEqual(fitted['maximum_remaining'],4)
        self.assertEqual([r['rows'] for r in fitted['trace']],[32,24,16,8])
        self.assertEqual(fitted['feature_indices'],[0,1,2,3,10,11])
        self.assertTrue(all(np.isfinite(r['teacher_rmse']) for r in fitted['trace']))
        first=fitted['trace'][0]
        np.testing.assert_allclose(first['target_sum'],(data['gross_log']-.25*data['breach']).sum(axis=0))
        import importlib.util
        spec=importlib.util.spec_from_file_location('nonzero_learning_replay',ROOT/'scripts/verify_learning.py')
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        self.assertEqual(verifier.verify_transitions(data,{1:scenario(rows=[(.1,.02,0.,0.)]*5)},config)[0],128)
        checked,_,error=verifier.verify_bellman({'train_ids':[1],'fqi':{'portfolio':{'0.25':fitted}}},data,config)
        self.assertEqual(checked,4)
        self.assertLess(error,1e-9)

    def test_immediate_fee_and_buffer_are_not_double_counted(self):
        x=np.zeros((1,len(FEATURE_NAMES)));x[0,:2]=[.5,.35];x[0,10]=.0025
        q=np.array([[0.,.0011,0.,0.]])
        config=dict(CONFIG,learning_buffer=0.)
        choice,net,benefit,fees=choose_actions(q,x,config)
        self.assertEqual(int(choice[0]),1)
        self.assertAlmostEqual(net[0],q[0,1]+np.log1p(-fees[0,1]))
        self.assertAlmostEqual(benefit[0,1],1-np.exp(-.0011))
        x[0,10]=.99
        self.assertEqual(int(choose_actions(q,x,config)[0][0]),0)


class LearnedDeploymentTests(unittest.TestCase):
    def test_horizon_is_required_and_paths_are_not_prefix_reused(self):
        with self.assertRaises(ValueError):FQIPolicy('fqi_student',manual_model(),dict(penalty=.25),CONFIG,None)
        path=scenario(10)
        metrics,records=evaluate_learning((path,),manual_model(),{'fqi_student':dict(penalty=.25)},CONFIG,
            names=['fqi_student'],horizons=[5,6])
        self.assertEqual(len(records),11)
        a=records[(records.evaluation_horizon==5)&(records.period==2)].iloc[0]
        b=records[(records.evaluation_horizon==6)&(records.period==2)].iloc[0]
        self.assertNotEqual(a.target_weights_equity,b.target_weights_equity)
        self.assertEqual(set(metrics.horizon),{5,6})

    def test_future_changes_cannot_change_prior_decisions(self):
        base=scenario(10,[(.1,0.,0.,.01)]*6)
        changed=replace(base,observations=base.observations[:4]+tuple(
            MarketObservation(10,t,(-.9,.2,0.),.2) for t in (5,6)))
        def run(path):
            policy=FQIPolicy('fqi_student',manual_model(),dict(penalty=.25),CONFIG,6)
            return simulate(path,policy,ANCHOR,SimulationConfig(6))
        a,b=run(base),run(changed)
        self.assertEqual(a.records[:4],b.records[:4])
        self.assertEqual(a.records[4].post_trade_holdings,b.records[4].post_trade_holdings)

    def test_review_is_absorbing_and_explanation_under_50_words(self):
        path=scenario(10,[(-.8,-.8,0.,0.)]+[(0.,0.,0.,0.)]*4)
        policy=FQIPolicy('fqi_student',manual_model(),dict(penalty=.25),CONFIG,5)
        result=simulate(path,policy,ANCHOR,SimulationConfig(5))
        self.assertEqual(result.records[1].action,Action.HUMAN_REVIEW)
        self.assertTrue(all(r.review_pending for r in result.records[1:]))
        self.assertTrue(all(r.transaction_cost==0 for r in result.records[1:]))
        self.assertLessEqual(len(result.records[1].explanation.split()),50)

    def test_training_overlap_and_final_freeze_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            evaluate_learning((scenario(1),),manual_model(),{},CONFIG,names=['balanced_annual'])
        with patch('safebench.experiment.load_partition',side_effect=AssertionError('Unexpected market read')):
            with self.assertRaisesRegex(ValueError,'development-only'):
                freeze(CONFIG,'unused','fqi_student','unused','unused')


class LearningStudyTests(unittest.TestCase):
    def scores(self):
        ref=pd.DataFrame([dict(policy='balanced_annual',scenario_id=i,real_cagr=.04,real_max_drawdown=.4) for i in range(1,9)])
        scores=pd.DataFrame([dict(policy=name,scenario_id=i,trial=trial,real_cagr=.01 if trial==0 else .04,
            real_max_drawdown=.2 if trial==0 else .35) for name in LEARNING_NAMES for i in range(1,9) for trial in range(2)])
        return scores,ref

    def test_risk_reduction_cannot_hide_growth_failure(self):
        scores,ref=self.scores()
        chosen,rows=select_learning_trials(scores,ref,CONFIG)
        self.assertTrue(all(v['selection_trial']==1 for v in chosen.values()))
        self.assertTrue(all(r['risk_feasible'] for r in rows if r['selected']))
        scores.real_cagr=.01
        chosen,_=select_learning_trials(scores,ref,CONFIG)
        self.assertTrue(all(v['selection_status']=='infeasible_diagnostic' for v in chosen.values()))

    def test_intervals_adjust_all_policies_endpoints_and_costs(self):
        rows=[dict(policy=n,scenario_id=i,horizon=30,cost_rate=c,real_cagr=.04,
            real_max_drawdown=.4 if n=='balanced_annual' else .3)
            for n in ['balanced_annual',*LEARNING_NAMES] for i in range(1,9) for c in (.0025,.005)]
        result=learning_intervals(pd.DataFrame(rows),dict(CONFIG,learning_bootstrap=100))
        self.assertEqual(len(result),18)
        self.assertTrue(result.passed.all())
        self.assertTrue((result.family_comparisons==18).all())

    def test_complete_generated_pipeline_audit_and_report(self):
        fixtures=tuple(scenario(sid) for sid in range(1,7))
        config=dict(CONFIG,outer_folds=3,inner_folds=2,learning_penalties=[.25],bootstrap=20,
                    learning_bootstrap=50,learning_teacher_depth=2,learning_student_depth=1,learning_min_leaf=2)
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            out=Path(td)/'generated'
            with patch('safebench.experiment.load_partition',return_value=(fixtures,'generated-only')):
                run_development(config,Path(td)/'never-read.csv',out)
            self.assertTrue(json.loads((out/'validation.json').read_text())['all_checks_passed'])
            self.assertEqual(len(list((out/'training_data').glob('*.npz'))),9)
            self.assertEqual(len(list((out/'figures').glob('*.pdf'))),8)
            self.assertEqual(len(pd.read_csv(out/'learning_suitability.csv')),15)
            self.assertEqual(len(pd.read_csv(out/'learning_intervals.csv')),18)
            self.assertIn('反事实离线强化学习',(out/'learning_report.html').read_text(encoding='utf-8'))
            import importlib.util
            spec=importlib.util.spec_from_file_location('independent_learning_replay',ROOT/'scripts/verify_learning.py')
            verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
            replay=verifier.verify_learning(out,fixtures)
            self.assertTrue(replay['all_checks_passed'])
            self.assertEqual(replay['training_datasets_checked'],9)
            self.assertEqual(replay['decision_rows_checked'],6*3*111)
            splits=json.loads((out/'splits.json').read_text())
            for split in splits:
                with np.load(out/'training_data'/f"outer_{split['outer_fold']}.npz",allow_pickle=False) as data:
                    self.assertTrue(set(data['scenario_id'])<=set(split['train_ids']))
                    self.assertTrue(set(data['scenario_id']).isdisjoint(split['validation_ids']))


if __name__=='__main__':unittest.main()
