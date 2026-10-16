"""Generated-fixture tests for causal training, policy gradients and ledger parity."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
import numpy as np
from safebench import ROOT
from safeinvest.contracts import MarketObservation, Scenario, SimulationConfig
from safebench.engine import simulate
from safebench.metrics import compute_metrics
from safebench.policies import fit_model, make_policy
from safebench.experiment import evaluate, select_experts
from safebench.experts import (EXPERT_NAMES, FEATURE_NAMES, Adam, features, forward, gradients,
    init_network, pack_network, unpack_network, training_arrays, rollout, fit_expert_bundle)

CONFIG=json.loads((ROOT/'configs/benchmark.json').read_text())


def generated_scenarios(n=8):
    rng=np.random.default_rng(731)
    scenarios=[]
    for sid in range(1,n+1):
        rows=[]
        for t in range(30):
            returns=(float(rng.normal(.065,.18)),float(rng.normal(.025,.04)),.015)
            rows.append(MarketObservation(sid,t+1,returns,float(rng.normal(.02,.008))))
        scenarios.append(Scenario(sid,tuple(rows)))
    return scenarios


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=deepcopy(CONFIG)
        cls.config['expert_research'].update(supervised_epochs=3,rl_epochs=3,drawdown_penalties=[.1])
        cls.ss=generated_scenarios()
        cls.model=fit_model(cls.ss[:6],cls.config)
        cls.model['experts']=fit_expert_bundle(cls.ss[:6],cls.model,cls.config)
        cls.params={name:{'drawdown_penalty':.1} for name in EXPERT_NAMES}

    def test_gradient_matches_finite_difference(self):
        rng=np.random.default_rng(8)
        net=init_network(rng,[4,3])
        x=rng.normal(size=(5,len(FEATURE_NAMES)))
        mask=np.array([[True,True,False]]*5)
        labels=np.array([0,1,0,1,1])
        p,cache=forward(net,x,mask)
        dz=p.copy(); dz[np.arange(5),labels]-=1
        dw,db=gradients(net,cache,dz/5)
        def loss():
            q,_=forward(net,x,mask)
            return -np.log(q[np.arange(5),labels]).mean()
        for array,grad in zip(net['weights']+net['biases'],dw+db):
            for index in np.ndindex(array.shape):
                old=array[index]; eps=1e-6
                array[index]=old+eps; plus=loss()
                array[index]=old-eps; minus=loss()
                array[index]=old
                self.assertAlmostEqual(grad[index],(plus-minus)/(2*eps),places=6)
        self.assertTrue((p[:,2]==0).all())

    def test_batch_training_simulator_matches_scalar_engine(self):
        real,hist=training_arrays(self.ss[6:])
        settings=dict(self.config['expert_research'],cost_rate=self.config['cost_rate'])
        fit=self.model['experts']['fits']['0.1']
        network=unpack_network(fit['networks']['moe_rl'][0])
        for cost in [0.,.0025,.01]:
            settings['cost_rate']=cost
            for h in [5,15,30]:
                _,rewards,objective,wealth,loss=rollout(real,hist,np.full(2,h),self.model,settings,.1,
                    np.random.default_rng(0),network,(np.array(fit['mean']),np.array(fit['scale'])),deterministic=True)
                for initial in [100.,1000.]:
                    cfg=dict(self.config,initial_wealth=initial)
                    for i,s in enumerate(self.ss[6:]):
                        p=make_policy('moe_rl_seed0',self.model,{'drawdown_penalty':.1},cfg,horizon=h)
                        m=compute_metrics(simulate(s,p,p.initial_target,SimulationConfig(h,initial,cost)))
                        self.assertAlmostEqual(m['terminal_real_wealth']/initial,wealth[i],places=12)
                        self.assertAlmostEqual(m['real_max_drawdown'],loss[i],places=12)
                        self.assertAlmostEqual(objective[i],np.log(wealth[i])/h-.1*loss[i],places=12)
                        initial_fee=cost*.85
                        self.assertAlmostEqual(rewards[:,i].sum()+np.log1p(-initial_fee)/h-.1*initial_fee,objective[i],places=12)

    def test_causal_prefix_and_benefit_gate(self):
        s=self.ss[-1]
        changed=replace(s,observations=s.observations[:10]+tuple(
            replace(o,nominal_returns=(-.6,.5,.1),inflation=.2) for o in s.observations[10:]))
        for name in EXPERT_NAMES:
            results=[]
            for path in [s,changed]:
                p=make_policy(name,self.model,self.params[name],self.config,horizon=30)
                results.append(simulate(path,p,p.initial_target,SimulationConfig(30)))
            self.assertEqual(results[0].records[:10],results[1].records[:10])
            self.assertEqual(results[0].records[10].action,results[1].records[10].action)
            for r in results[0].records[1:]:
                if r.moved_amount>0:
                    self.assertGreater(r.estimated_benefit,r.estimated_cost+p.buffer)
                    self.assertLessEqual(len(r.explanation.split()),50)

    def test_fit_isolation_and_repeatability(self):
        with self.assertRaises(PermissionError):
            fit_expert_bundle([replace(self.ss[0],scenario_id=121)],self.model,self.config)
        with self.assertRaises(ValueError):
            evaluate(self.ss[:1],self.model,self.params,self.config,names=['moe_rl'])
        other=fit_expert_bundle(self.ss[:6],self.model,self.config)
        self.assertEqual(other['fits']['0.1']['networks'],self.model['experts']['fits']['0.1']['networks'])
        bad=deepcopy(self.model); bad['experts']['train_ids']=[120]
        with self.assertRaises(ValueError):
            make_policy('moe_rl',bad,self.params['moe_rl'],self.config,horizon=30)

    def test_horizons_have_distinct_simulations(self):
        m,r=evaluate(self.ss[6:],self.model,self.params,self.config,names=['moe_rl'],horizons=[5,30])
        self.assertEqual(len(m),4)
        self.assertEqual(len(r),2*(5+30))
        self.assertFalse(r.duplicated(['policy','scenario_id','simulation_horizon','period','cost_rate']).any())
        for h in [5,30]:
            p=make_policy('moe_rl',self.model,self.params['moe_rl'],self.config,horizon=h)
            direct=compute_metrics(simulate(self.ss[-1],p,p.initial_target,SimulationConfig(h)))
            saved=m[(m.scenario_id==8)&(m.horizon==h)].iloc[0]
            self.assertAlmostEqual(direct['terminal_real_wealth'],saved.terminal_real_wealth)
        with self.assertRaises(ValueError):
            make_policy('moe_rl',self.model,self.params['moe_rl'],self.config)

    def test_network_serialization_and_static_ablation(self):
        net=init_network(np.random.default_rng(1),[3],static=True)
        x=np.ones((2,len(FEATURE_NAMES))); x[1]*=5
        mask=np.ones((2,3),dtype=bool)
        p,_=forward(unpack_network(pack_network(net)),x,mask)
        np.testing.assert_array_equal(p[0],p[1])

    def test_invalid_estimate_routes_to_persistent_review(self):
        from safeinvest.contracts import Action
        bad=deepcopy(self.model); bad['mu']=[float('nan')]*3
        p=make_policy('moe_rl',bad,self.params['moe_rl'],self.config,horizon=5)
        result=simulate(self.ss[-1],p,p.initial_target,SimulationConfig(5))
        self.assertEqual(result.records[1].action,Action.HUMAN_REVIEW)
        self.assertEqual(result.records[2].reason_codes,('REVIEW_PENDING',))
        self.assertTrue(all(r.transaction_cost==0 for r in result.records[1:]))
        with self.assertRaisesRegex(ValueError,'Invalid utility'):
            fit_expert_bundle(self.ss[:6],bad,self.config)

    def test_no_feasible_trial_is_explicit(self):
        import pandas as pd
        rows=[]
        for name in ['balanced_annual',*EXPERT_NAMES]:
            for sid in range(1,11):
                rows.append(dict(policy=name,trial=0,scenario_id=sid,
                    real_cagr=.05 if name=='balanced_annual' else .01,real_max_drawdown=.2))
        chosen,records=select_experts(pd.DataFrame(rows),self.config)
        self.assertEqual(set(chosen),set(EXPERT_NAMES))
        self.assertTrue(all(not r['risk_selection_feasible'] for r in records))

    def test_expert_freeze_and_final_use_generated_data_only(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        from safebench.data import load_partition
        from safebench.experiment import freeze, run_final, manifest, code_digest
        config=deepcopy(self.config)
        config.update(inner_folds=2,tuning_grid=[CONFIG['tuning_grid'][0]],bootstrap=20,sensitivity_costs=[.005])
        config['expert_research'].update(supervised_epochs=1,rl_epochs=1)
        config['frontier_research']={'enabled':False}
        config['cushion_research']={'enabled':False}
        config['calibrated_research']={'enabled':False}
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            root=Path(td); dev=root/'dev'; dev.mkdir(); data=root/'generated.csv'
            with data.open('w') as f:
                f.write('scenario_id,period,equity_return,bond_return,bill_proxy_return,inflation\n')
                for sid in range(1,181):
                    for t in range(1,31): f.write(f'{sid},{t},0,0,0,0\n')
            _,sha=load_partition(data)
            (dev/'manifest.json').write_text(json.dumps(manifest(config,sha,list(range(1,121)),'development')))
            pd.DataFrame([dict(profile_id='C',horizon=30,loss_tolerance=.4,liquidity='low',
                               portfolio='balanced',policy='moe_rl',explanation='fixture')]).to_csv(dev/'suitability.csv',index=False)
            bundle=root/'bundle.json'
            freeze(config,data,'moe_rl',bundle,dev)
            frozen=json.loads(bundle.read_text())
            self.assertEqual(frozen['model']['experts']['train_ids'],list(range(1,121)))
            sha=code_digest()
            with patch('safebench.experiment.BASE',root),patch('safebench.experiment.code_digest',return_value=sha),patch('safebench.report.generate_report'):
                run_final(bundle,data,root/'final',True)
            metrics=pd.read_csv(root/'final/scenario_metrics.csv')
            self.assertEqual(set(metrics.scenario_id),set(range(121,181)))
            self.assertIn('moe_rl',set(metrics.policy))
            assignments=pd.read_csv(root/'final/suitability.csv')
            self.assertEqual(assignments.iloc[0].policy,'moe_rl')
            group=metrics[(metrics.policy=='moe_rl')&(metrics.horizon==30)&np.isclose(metrics.cost_rate,.0025)]
            self.assertAlmostEqual(assignments.iloc[0].final_median_growth,group.real_cagr.median())
            self.assertTrue(json.loads((root/'final/validation.json').read_text())['all_checks_passed'])


if __name__=='__main__':
    unittest.main()
