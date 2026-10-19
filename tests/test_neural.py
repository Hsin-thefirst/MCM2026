"""Generated-market and finite-difference checks for constrained neural path training."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from safebench import ROOT
from safeinvest.contracts import Scenario,MarketObservation,SimulationConfig,Action
from safebench.engine import simulate
from safebench.metrics import compute_metrics,expected_shortfall
from safebench.neural_core import neural_data,normalizer,neural_target,NEURAL_NAMES,NeuralPolicy
from safebench.neural_training import (torch_runtime,tensor_data,network,rollout,restore,tail_mean,fit_neural_models)
from safebench.neural_study import select_neural_trials
from safebench.policies import fit_model,make_policy
from safebench.experiment import run_development,freeze

CONFIG=json.loads((ROOT/'configs/neural.json').read_text())
TORCH_AVAILABLE=importlib.util.find_spec('torch') is not None


def scenario(sid=1,zero=False):
    rows=[(0.,0.,0.,0.) if zero else (.09+.002*sid if t%3 else -.13-.001*sid,.018,.01,.02)
          for t in range(1,31)]
    return Scenario(sid,tuple(MarketObservation(sid,t,tuple(row[:3]),row[3]) for t,row in enumerate(rows,1)))


def verifier():
    spec=importlib.util.spec_from_file_location('independent_neural',ROOT/'scripts/verify_neural.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class NeuralDataTests(unittest.TestCase):
    def test_training_ids_sequence_and_returns_are_checked(self):
        for paths in ((scenario(121),),(scenario(),scenario())):
            with self.assertRaises(PermissionError):neural_data(paths,CONFIG)
        base=scenario()
        bad=replace(base,observations=(replace(base.observations[0],inflation=-1.),)+base.observations[1:])
        with self.assertRaises(ValueError):neural_data((bad,),CONFIG)

    def test_market_features_are_prefix_only_and_independent_replay_agrees(self):
        base=scenario();changed=replace(base,observations=base.observations[:10]+tuple(
            replace(o,nominal_returns=(-.8,.3,.01)) for o in base.observations[10:]))
        a,b=neural_data((base,),CONFIG),neural_data((changed,),CONFIG)
        np.testing.assert_array_equal(a['history'][:,:11],b['history'][:,:11])
        expected=verifier().market_arrays((base,))
        for key in a:np.testing.assert_allclose(a[key],expected[key],atol=1e-14)
        mean,scale=normalizer(a,CONFIG)
        self.assertTrue(np.isfinite(mean).all() and np.all(scale>0))

    def test_freeze_rejected_before_any_market_read(self):
        with patch('safebench.experiment.load_partition',side_effect=AssertionError('Unexpected market read')):
            with self.assertRaisesRegex(ValueError,'development-only'):
                freeze(CONFIG,'unused','neural_mlp','unused','unused')


@unittest.skipUnless(TORCH_AVAILABLE,'Install the neural extra to run automatic-differentiation tests')
class NeuralTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=dict(CONFIG,neural_epochs=2,neural_checkpoints=[1,2],neural_hidden=4,neural_seeds=[17])
        cls.paths=tuple(scenario(i) for i in range(1,5))
        cls.model=fit_model(cls.paths,cls.config)

    def tensors(self,data,fit):
        torch=torch_runtime()
        return (tensor_data(data),torch.tensor(fit['mean'],dtype=torch.float64),torch.tensor(fit['scale'],dtype=torch.float64),
                torch.tensor(self.model['mu'],dtype=torch.float64),torch.tensor(self.model['cov'],dtype=torch.float64))

    def test_tail_mean_handles_fractional_mass(self):
        torch=torch_runtime();values=torch.tensor([1.,2.,3.,4.,5.,6.,7.],dtype=torch.float64)
        self.assertAlmostEqual(float(tail_mean(values,.8)),expected_shortfall(values.numpy(),.8))

    def test_tensor_numpy_and_engine_match_for_horizons_costs_and_networks(self):
        torch=torch_runtime();path=scenario(10);data=neural_data((path,),self.config);independent=verifier()
        for name in NEURAL_NAMES:
            fit=self.model['neural'][name];nets=[restore(m) for m in fit['checkpoints']['2']['members']]
            tensors=self.tensors(data,fit)
            for horizon in (5,6):
                with torch.no_grad():values=rollout(nets,name,*tensors,self.config,horizon)
                for slot,cost in enumerate((.0025,.005)):
                    policy=make_policy(name,self.model,dict(checkpoint=2),self.config,horizon=horizon)
                    result=simulate(path,policy,policy.initial_target,SimulationConfig(horizon,100.,cost))
                    metric=compute_metrics(result)
                    other=independent.numpy_rollout(fit,2,self.model,data,self.config,horizon,cost)
                    for key,tensor in [('terminal_real_wealth','wealth'),('real_max_drawdown','mdd'),
                                       ('total_real_cost','total_real_cost'),('trade_rate','trade_rate')]:
                        self.assertAlmostEqual(metric[key],float(values[tensor][slot]),places=9)
                        self.assertAlmostEqual(metric[key],other[tensor][0],places=9)
                    self.assertEqual([r.action.value for r in result.records],other['actions'][0].tolist())

    def test_path_gradient_agrees_with_finite_difference(self):
        torch=torch_runtime();name='neural_linear';fit=self.model['neural'][name]
        tensors=list(self.tensors(neural_data(self.paths,self.config),fit))
        tensors[3]=torch.zeros(3,dtype=torch.float64)
        tensors[4]=torch.diag(torch.tensor([.06,.002,.0001],dtype=torch.float64))
        net=network(name,self.config,17)
        def objective():
            values=rollout([net],name,*tensors,self.config,5)
            return -values['log_growth'].mean()+.3*tail_mean(values['mdd'])
        loss=objective();loss.backward();observed=float(net[-1].bias.grad[0]);epsilon=1e-6
        with torch.no_grad():net[-1].bias[0]+=epsilon
        plus=float(objective().detach())
        with torch.no_grad():net[-1].bias[0]-=2*epsilon
        minus=float(objective().detach())
        self.assertGreater(abs(observed),1e-7)
        self.assertAlmostEqual(observed,(plus-minus)/(2*epsilon),places=6)

    def test_checkpoint_constraints_seeds_and_legal_targets_are_preserved(self):
        x=np.zeros((3,13));x[:,:2]=[.5,.35]
        for name,fit in self.model['neural'].items():
            self.assertEqual(fit['train_ids'],[1,2,3,4]);self.assertEqual(fit['seeds'],[17])
            self.assertEqual(set(fit['checkpoints']),{'1','2'})
            self.assertEqual(len(fit['checkpoints']['2']['training_constraints']),10)
            target=neural_target(fit,2,x)
            np.testing.assert_allclose(target.sum(axis=1),1,atol=1e-12)
            self.assertTrue((target[:,0]<=.5+1e-12).all() and (target>=0).all())
            self.assertTrue(all(np.isfinite(t['gradient_norm']) for t in fit['trace']))

    def test_future_returns_cannot_change_frozen_decisions(self):
        base=scenario(10);changed=replace(base,observations=base.observations[:4]+tuple(
            replace(o,nominal_returns=(-.8,.5,.1)) for o in base.observations[4:]))
        def run(path):
            p=make_policy('neural_mlp',self.model,dict(checkpoint=2),self.config,horizon=6)
            return simulate(path,p,p.initial_target,SimulationConfig(6))
        a,b=run(base),run(changed)
        self.assertEqual(a.records[:4],b.records[:4]);self.assertEqual(a.records[4].post_trade_holdings,b.records[4].post_trade_holdings)

    def test_horizon_required_and_review_absorbing(self):
        with self.assertRaises(ValueError):NeuralPolicy('neural_mlp',self.model,dict(checkpoint=2),self.config,None)
        path=scenario(10);path=replace(path,observations=(replace(path.observations[0],nominal_returns=(-.8,-.8,0.)),)+path.observations[1:])
        p=make_policy('neural_mlp',self.model,dict(checkpoint=2),self.config,horizon=5)
        result=simulate(path,p,p.initial_target,SimulationConfig(5))
        self.assertEqual(result.records[1].action,Action.HUMAN_REVIEW)
        self.assertTrue(all(r.review_pending for r in result.records[1:]))
        self.assertTrue(all(r.transaction_cost==0 for r in result.records[1:]))
        self.assertLessEqual(len(result.records[1].explanation.split()),50)


class NeuralSelectionTests(unittest.TestCase):
    def test_checkpoint_selection_does_not_trade_growth_for_lower_risk_silently(self):
        ref=pd.DataFrame([dict(scenario_id=i,real_cagr=.04,real_max_drawdown=.4) for i in range(1,9)])
        scores=pd.DataFrame([dict(policy=n,scenario_id=i,trial=t,real_cagr=.01 if t==0 else .04,
            real_max_drawdown=.2 if t==0 else .35) for n in NEURAL_NAMES for i in range(1,9) for t in range(2)])
        chosen,rows=select_neural_trials(scores,ref,CONFIG)
        self.assertTrue(all(p['checkpoint']==120 for p in chosen.values()))
        self.assertTrue(all(r['risk_feasible'] for r in rows if r['selected']))
        scores.real_cagr=.01;chosen,_=select_neural_trials(scores,ref,CONFIG)
        self.assertTrue(all(p['selection_status']=='infeasible_diagnostic' for p in chosen.values()))

    @unittest.skipUnless(TORCH_AVAILABLE,'Install neural extra')
    def test_generated_pipeline_and_independent_training_constraint_replay(self):
        fixtures=tuple(scenario(i,zero=True) for i in range(1,7))
        config=dict(CONFIG,outer_folds=3,inner_folds=2,neural_epochs=2,neural_checkpoints=[1,2],
            neural_hidden=4,neural_seeds=[17],bootstrap=20,neural_bootstrap=50)
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            out=Path(td)/'generated'
            with patch('safebench.experiment.load_partition',return_value=(fixtures,'generated-only')):
                run_development(config,Path(td)/'never-read.csv',out)
            self.assertTrue(json.loads((out/'validation.json').read_text())['all_checks_passed'])
            replay=verifier().verify_neural(out,fixtures)
            self.assertTrue(replay['all_checks_passed']);self.assertEqual(replay['training_datasets_checked'],9)
            self.assertEqual(replay['decision_rows_checked'],6*3*111)
            self.assertEqual(len(list((out/'training_data').glob('*.npz'))),9)
            self.assertEqual(len(pd.read_csv(out/'neural_intervals.csv')),18)
            self.assertEqual(len(list((out/'figures').glob('*.pdf'))),8)
            self.assertIn('完整路径风险约束',(out/'neural_report.html').read_text(encoding='utf-8'))
            self.assertFalse(pd.read_csv(out/'neural_screen.csv').all_checks_passed.any())


if __name__=='__main__':unittest.main()
