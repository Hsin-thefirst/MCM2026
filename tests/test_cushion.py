"""Causal, cost-aware checks for the fine-grid risk preference policy."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
import numpy as np
from safebench import ROOT
from safeinvest.contracts import SimulationConfig,PolicyContext,Action
from safebench.engine import simulate
from safebench.policies import fit_model,make_policy
from safebench.cushion import CUSHION_NAMES,CALIBRATED_NAMES
from test_experts import generated_scenarios


class CushionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'configs/benchmark.json').read_text())
        cls.basic=dict(cls.config,frontier_research={'enabled':False},expert_research={'enabled':False})
        cls.ss=generated_scenarios()
        cls.model=fit_model(cls.ss[:6],cls.basic)

    def test_causality_and_exact_cost_gate(self):
        s=self.ss[-1]
        changed=replace(s,observations=s.observations[:10]+tuple(replace(o,nominal_returns=(-.8,.4,.01)) for o in s.observations[10:]))
        for name in CUSHION_NAMES+CALIBRATED_NAMES:
            for budget in [.4,.5]:
                results=[]
                for path in [s,changed]:
                    p=make_policy(name,self.model,{'loss_budget':budget},self.config,horizon=30)
                    results.append(simulate(path,p,p.initial_target,SimulationConfig(30)))
                self.assertEqual(results[0].records[:10],results[1].records[:10])
                self.assertEqual(results[0].records[10].action,results[1].records[10].action)
                for r in results[0].records[1:]:
                    if r.moved_amount:
                        self.assertGreater(r.estimated_benefit,r.estimated_cost+p.buffer)
                        self.assertAlmostEqual(r.estimated_cost,r.transaction_cost/sum(r.pre_trade_holdings))
                        self.assertLessEqual(len(r.explanation.split()),50)

    def test_risk_aversion_increases_and_saturates(self):
        p=make_policy('cushion_gate',self.model,{'loss_budget':.4},self.config,horizon=30)
        values=[]
        for dd in [0,.1,.2,.3,.38,.45,.55]:
            c=PolicyContext(2,self.ss[-1].observations[:1],tuple((1-dd)*100*np.array(p.initial_target)),
                p.initial_target,1.,100.,False,.0025,dd)
            decision=p.decide(c)
            code=next(c for c in decision.reason_codes if c.startswith('RISK_AVERSION_'))
            values.append(float(code.split('_')[-1]))
        self.assertEqual(values[0],4.)
        self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))
        self.assertEqual(values[-1],values[-2])

    def test_invalid_moments_suspend_without_fictitious_liquidation(self):
        model=deepcopy(self.model); model['mu'][0]=float('nan')
        p=make_policy('cushion_gate',model,{'loss_budget':.4},self.config,horizon=5)
        result=simulate(self.ss[-1],p,p.initial_target,SimulationConfig(5))
        self.assertEqual(result.records[1].action,Action.HUMAN_REVIEW)
        self.assertEqual(result.records[2].reason_codes,('REVIEW_PENDING',))
        self.assertTrue(all(r.moved_amount==0 for r in result.records[1:]))

    def test_strategic_prior_has_balanced_optimum_at_peak(self):
        p=make_policy('cushion_static',self.model,{'loss_budget':.5},self.config,horizon=30)
        gamma=self.config['cushion_research']['gamma_at_peak']
        values=p.candidates@p.mu-.5*gamma*np.einsum('ni,ij,nj->n',p.candidates,p.cov,p.candidates)
        np.testing.assert_allclose(p.candidates[int(values.argmax())],p.initial_target,atol=1e-12)
        self.assertAlmostEqual(np.array(p.initial_target)@p.mu,
                               np.array(p.initial_target)@np.array(self.model['mu']),places=12)

    def test_calibrated_policy_uses_empirical_moments_and_selected_gamma(self):
        p=make_policy('calibrated_gate',self.model,{'gamma':4.5},self.config,horizon=30)
        np.testing.assert_array_equal(p.mu,self.model['mu'])
        self.assertEqual(p.gamma_at_peak,4.5)
        c=PolicyContext(2,self.ss[-1].observations[:1],(45.,30.,25.),p.initial_target,1.,100.,False,.0025,.1)
        result=p.decide(c)
        self.assertIn('RISK_AVERSION_4.5000',result.reason_codes)


if __name__=='__main__': unittest.main()
