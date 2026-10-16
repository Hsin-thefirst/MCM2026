"""Analytical checks of path risk, causal decisions and deployment assignments."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from safebench import ROOT
from safeinvest.contracts import Action, Decision, SimulationConfig
from safebench.engine import simulate
from safebench.experiment import evaluate, suitability
from safebench.frontier import FRONTIER_NAMES, annual_paths, value_estimates, tail_mean
from safebench.metrics import compute_metrics, expected_shortfall
from safebench.policies import fit_model, make_policy, StaticPolicy
from test_experts import generated_scenarios


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'configs/benchmark.json').read_text())
        cls.config['frontier_research'].update(training_states=40,epochs=3)
        cls.ss=generated_scenarios()
        cls.model=fit_model(cls.ss[:6],cls.config)

    def test_annual_grid_matches_scalar_ledger_at_every_horizon(self):
        real=np.array([[[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns]
                        for o in s.observations] for s in self.ss[6:]])
        targets=np.array([[.3,.6,.1],[.5,.35,.15],[0,0,1]])
        for fee in [0.,.0025,.01]:
            growth,loss=annual_paths(real,targets,fee)
            for i,target in enumerate(targets):
                for j,s in enumerate(self.ss[6:]):
                    for h in [5,6,15,25,30]:
                        p=StaticPolicy('fixed',tuple(target))
                        m=compute_metrics(simulate(s,p,tuple(target),SimulationConfig(h,100,fee)))
                        self.assertAlmostEqual(m['real_cagr'],growth[h-1,i,j],places=12)
                        self.assertAlmostEqual(m['real_max_drawdown'],loss[h-1,i,j],places=12)

    def test_multi_period_value_matches_manual_counterfactual(self):
        gross=np.array([[[.7,1.1,1.0],[.9,1.2,1.02]],[[1.1,.9,1.0],[.8,1.3,1.01]]])
        w=np.array([[.5,.3,.2]]); targets=np.array([[[.5,.3,.2],[.2,.3,.5],[.3,.6,.1]]])
        q,fees=value_estimates(targets,w,np.array([.1]),np.array([.2]),np.array([2]),
                              np.array([5]),gross,.3,.01)
        for a,target in enumerate(targets[0]):
            terminal=[]; losses=[]
            for path in gross:
                wealth=1-fees[0,a]; peak=1/.9; maximum=max(.2,1-wealth/peak)
                for period in path:
                    value=wealth*float(target@period)
                    peak=max(peak,value); maximum=max(maximum,1-value/peak)
                terminal.append(value); losses.append(maximum)
            expected=np.mean(np.log(terminal))/5-.3*(expected_shortfall(losses)-.2)
            self.assertAlmostEqual(q[0,a],expected,places=12)
        for n in [1,3,6,11,32]:
            a=np.arange(n,dtype=float)
            self.assertAlmostEqual(tail_mean(a),expected_shortfall(a))

    def test_matched_controls_share_mpc_parameters(self):
        cfg=deepcopy(self.config)
        params={'frontier_mpc':{'frontier_trial':0},'frontier_hold':{'frontier_trial':1}}
        m,_=evaluate(self.ss[6:],self.model,params,cfg,names=['frontier_matched_hold'],horizons=[30])
        for s in self.ss[6:]:
            p=make_policy('frontier_hold',self.model,{'frontier_trial':0},cfg,horizon=30)
            expected=compute_metrics(simulate(s,p,p.initial_target,SimulationConfig(30)))
            actual=m[m.scenario_id==s.scenario_id].iloc[0]
            self.assertAlmostEqual(actual.terminal_real_wealth,expected['terminal_real_wealth'])

    def test_risk_barrier_uses_same_state_strategic_reference(self):
        gross=np.array([[[.5,1.1,1.]],[[.8,1.1,1.]],[[1.1,.8,1.]]])
        weights=np.array([[.7,.2,.1]])
        targets=np.array([[[.7,.2,.1],[.4,.4,.2],[.2,.3,.5]]])
        args=(targets,weights,np.zeros(1),np.zeros(1),np.ones(1,dtype=int),np.full(1,5),gross,.01,.0025)
        ordinary,fees=value_estimates(*args)
        guarded,_=value_estimates(*args,risk_barrier=.5,risk_slack=.01)
        losses=np.maximum(0,1-(np.einsum('bak,stk->bast',targets,gross)*(1-fees)[:,:,None,None]))
        p90=np.quantile(losses.max(axis=3),.9,axis=2)
        np.testing.assert_allclose(ordinary-guarded,.5*np.maximum(0,p90-p90[:,1:2]-.01))
        self.assertEqual(ordinary[0,1],guarded[0,1])

    def test_identical_asset_returns_do_not_justify_switching(self):
        model=deepcopy(self.model)
        model['frontier']['gross_paths']=[[[1.03**(t+1)]*3 for t in range(30)] for _ in range(6)]
        model['frontier']['annual_returns']=[[.03]*3 for _ in range(180)]
        for name in ['frontier_gate','frontier_mpc']:
            for cost in [0.,.0025,.01]:
                p=make_policy(name,model,{'frontier_trial':0},self.config,horizon=5)
                result=simulate(self.ss[-1],p,p.initial_target,SimulationConfig(5,100,cost))
                self.assertTrue(all(r.action==Action.KEEP for r in result.records[1:]))

    def test_fee_sensitivity_keeps_untrained_feature_constant(self):
        from unittest.mock import patch
        from safeinvest.contracts import PolicyContext
        captured=[]
        def network(net,x):
            captured.append(x.copy())
            return np.zeros((len(x),3)),[]
        p=make_policy('frontier_neural',self.model,{'frontier_trial':0},self.config,horizon=30)
        c=PolicyContext(2,self.ss[-1].observations[:1],(45.,30.,25.),p.initial_target,1.,100.,False,.0025,.0)
        with patch('safebench.frontier.network_values',side_effect=network):
            p.decide(c); p.decide(replace(c,transaction_cost_rate=.01))
        for raw in captured[1:]:
            np.testing.assert_array_equal(captured[0],raw)

    def test_economic_suspension_is_persistent_and_does_not_liquidate(self):
        s=self.ss[-1]
        bad=replace(s,observations=(replace(s.observations[0],nominal_returns=(-.8,-.8,-.8)),)+s.observations[1:])
        for name in ['frontier_hold','frontier_gate','frontier_mpc','frontier_linear','frontier_neural']:
            p=make_policy(name,self.model,{'frontier_trial':0},self.config,horizon=5)
            result=simulate(bad,p,p.initial_target,SimulationConfig(5))
            self.assertEqual(result.records[1].action,Action.HUMAN_REVIEW)
            self.assertEqual(result.records[2].reason_codes,('REVIEW_PENDING',))
            self.assertTrue(all(r.moved_amount==0 for r in result.records[1:]))

    def test_running_maximum_drawdown_survives_recovery(self):
        seen=[]
        class Spy:
            name='spy'
            def decide(self,c):
                seen.append(c)
                return Decision(Action.KEEP,('HOLD',),'Hold.')
        from safeinvest.contracts import MarketObservation,Scenario
        s=Scenario(1,tuple(MarketObservation(1,t+1,(r,r,r),0.) for t,r in enumerate([-.3,1.,0.])))
        simulate(s,Spy(),(1.,0.,0.),SimulationConfig(3,100,0))
        self.assertAlmostEqual(seen[-1].real_max_drawdown,.3)
        self.assertAlmostEqual(1-sum(seen[-1].holdings)/seen[-1].real_peak,0)

    def test_causal_prefix_gate_and_review(self):
        s=self.ss[-1]
        changed=replace(s,observations=s.observations[:8]+tuple(
            replace(o,nominal_returns=(-.8,.3,.01)) for o in s.observations[8:]))
        for name in FRONTIER_NAMES:
            runs=[]
            for path in (s,changed):
                p=make_policy(name,self.model,{'frontier_trial':0},self.config,horizon=15)
                runs.append(simulate(path,p,p.initial_target,SimulationConfig(15)))
            self.assertEqual(runs[0].records[:8],runs[1].records[:8])
            self.assertEqual(runs[0].records[8].action,runs[1].records[8].action)
            for r in runs[0].records[1:]:
                if p.requires_benefit_gate and r.moved_amount>0:
                    self.assertGreater(r.estimated_benefit,r.estimated_cost+p.buffer)
                    self.assertLessEqual(len(r.explanation.split()),50)
        bad=deepcopy(self.model)
        bad['frontier']['gross_paths'][0][0][0]=float('nan')
        p=make_policy('frontier_mpc',bad,{'frontier_trial':0},self.config,horizon=5)
        run=simulate(s,p,p.initial_target,SimulationConfig(5))
        self.assertEqual(run.records[1].action,Action.HUMAN_REVIEW)
        self.assertEqual(run.records[2].reason_codes,('REVIEW_PENDING',))

    def test_customer_mapping_uses_actual_balanced_policy(self):
        rows=[]
        for name in ['conservative_annual','growth_annual','balanced_annual','frontier_mpc']:
            for h in self.config['horizons']:
                for sid in range(1,121):
                    rows.append(dict(policy=name,horizon=h,scenario_id=sid,cost_rate=.0025,
                        real_max_drawdown=.05 if name=='frontier_mpc' else .8,real_cagr=.04))
        assigned,risk=suitability(pd.DataFrame(rows),self.config,balanced_policy='frontier_mpc')
        customer=assigned[assigned.profile_id=='C'].iloc[0]
        self.assertEqual(customer.policy,'frontier_mpc')
        self.assertEqual(customer.portfolio,'balanced')
        self.assertTrue((risk[risk.portfolio=='balanced'].policy=='frontier_mpc').all())

    def test_complete_grid_audit_rejects_entire_missing_path(self):
        from safebench.validation import audit_outputs
        names=['frontier_hold','frontier_mpc']
        cfg=deepcopy(self.config); cfg['horizons']=[5]; cfg['sensitivity_costs']=[]
        m,r=evaluate(self.ss[6:],self.model,{},cfg,names=names)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            (out/'config.json').write_text(json.dumps(cfg))
            (out/'manifest.json').write_text(json.dumps(dict(phase='development',scenario_ids=[7,8],evaluated_policies=names)))
            m.to_csv(out/'scenario_metrics.csv',index=False); r.to_csv(out/'decisions.csv.gz',index=False)
            self.assertTrue(audit_outputs(out)['all_checks_passed'])
            m=m[~((m.policy=='frontier_hold')&(m.scenario_id==7))]
            r=r[~((r.policy=='frontier_hold')&(r.scenario_id==7))]
            m.to_csv(out/'scenario_metrics.csv',index=False); r.to_csv(out/'decisions.csv.gz',index=False)
            with self.assertRaisesRegex(AssertionError,'grid'):
                audit_outputs(out)


if __name__=='__main__':
    unittest.main()
