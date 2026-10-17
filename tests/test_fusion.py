"""Independent scalar value and causal execution tests for research adapters."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fusion_policy import FusionPolicy, hold_values
from run_fusion_research import scalar_check
from test_plan_bc import fixture
from safeinvest.contracts import PolicyContext, SimulationConfig
from safebench.engine import simulate
from safebench.metrics import expected_shortfall


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.scenarios,self.config,self.model,self.params = fixture()
        self.settings = json.loads((Path(__file__).resolve().parents[1]/'configs/fusion_research.json').read_text())

    def test_value_matches_scalar_with_fee_marks_and_historical_loss(self):
        future = np.array(self.model['scenario_control']['real_returns'])[:, 3:5]
        targets = np.array([[.7,.2,.1],[.5,.3,.2],[0,0,1.]])
        q,fees,risk = hold_values(targets,targets[0],.12,.35,future,5,.005,self.settings)
        scalar_growth=[]; scalar_losses=[]
        for target in targets:
            fee=.005*sum(abs(float(a-b)) for a,b in zip(target,targets[0]))/2
            losses=[]; logs=[]
            for path in future:
                wealth=1-fee; peak=1/.88; maximum=max(.35,1-wealth/peak)
                holdings=[wealth*x for x in target]
                for returns in path:
                    holdings=[v*(1+r) for v,r in zip(holdings,returns)]
                    wealth=sum(holdings); maximum=max(maximum,1-wealth/peak); peak=max(peak,wealth)
                logs.append(np.log(wealth)); losses.append(maximum)
            scalar_growth.append(np.mean(logs)/5); scalar_losses.append(losses)
        p90=np.quantile(scalar_losses,.9,axis=1)
        sr=np.array([expected_shortfall(x) for x in scalar_losses])
        sq=np.array(scalar_growth)-.03*(sr-.35)-.5*np.maximum(0,p90-p90[1]-.01)
        np.testing.assert_allclose(q,sq,atol=1e-14)
        np.testing.assert_allclose(risk,sr,atol=1e-14)

    def test_all_variants_causal_capped_and_accounted(self):
        s=replace(self.scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in self.scenarios[0].observations))
        changed=replace(s,observations=s.observations[:3]+tuple(replace(o,nominal_returns=(2.,-.5,0.)) for o in s.observations[3:]))
        for h in (5,30):
            for name in self.settings['policies']:
                def policy(): return FusionPolicy(name,self.model,self.params,self.config,h,self.settings)
                p=policy(); result=simulate(s,p,p.initial_target,SimulationConfig(h,100.,.005))
                other=policy(); altered=simulate(changed,other,other.initial_target,SimulationConfig(h,100.,.005))
                scalar_check(result,s)
                for a,b in zip(result.records[:4],altered.records[:4]):
                    self.assertEqual(a.action,b.action)
                    self.assertEqual(a.post_trade_holdings,b.post_trade_holdings)
                for r in result.records:
                    if r.policy_diagnostics:
                        d=r.policy_diagnostics; selected=d['chosen']
                        self.assertGreaterEqual(d['values'][selected],d['values'][0]-1e-14)

    def test_action_ablation_and_no_initial_allocation_change(self):
        c=PolicyContext(2,self.scenarios[0].observations[:1],(42.5,25.5,17.),(.5,.3,.2),1.,100.,False,.0025)
        arrays=[]
        for name in self.settings['policies']:
            p=FusionPolicy(name,self.model,self.params,self.config,5,self.settings)
            self.assertEqual(p.initial_target,(.5,.3,.2))
            from safebench.plan_a import PlanAPolicy
            advice=PlanAPolicy._decide(p,c)
            candidates=p.candidates(c,advice); arrays.append(candidates)
            self.assertTrue(np.all(.5*np.abs(candidates-np.array(c.holdings)/85).sum(axis=1)<=.1+1e-12))
        np.testing.assert_array_equal(arrays[0],arrays[1][:4])
        np.testing.assert_array_equal(arrays[0],arrays[2][:4])
        self.assertLessEqual(arrays[2][-1,0],arrays[1][-1,0])

    def test_reject_final_training_and_preserve_review(self):
        p=FusionPolicy('fusion_mpc',self.model,self.params,self.config,5,self.settings)
        c=PolicyContext(2,self.scenarios[0].observations[:1],(10.,5.,5.),(.5,.3,.2),1.,100.,False,.0025)
        self.assertEqual(p.decide(c).action.value,'human_review')
        import copy
        bad=copy.deepcopy(self.model)
        for obj in (bad,bad['plan_a'],bad['scenario_control']): obj['train_ids']=[121]
        with self.assertRaises(ValueError): FusionPolicy('fusion_mpc',bad,self.params,self.config,5,self.settings)
