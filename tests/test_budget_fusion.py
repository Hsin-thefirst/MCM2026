"""Explicit resampling, units, and causal gates for the last fixed risk rules."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from budget_fusion import BudgetFusion,budget_values,bootstrap_log_risk
from fusion_value_extensions import uncertainty_values,hold_paths
from fusion_policy import hold_values
from safebench.metrics import expected_shortfall
from safebench.engine import simulate
from safebench.plan_a import PlanAPolicy
from safeinvest.contracts import SimulationConfig
from test_plan_bc import fixture


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.scenarios,self.config,self.model,self.params=fixture()
        self.settings=json.loads((ROOT/'configs/fusion_research.json').read_text())
        self.rule=dict(buffer=.02,short_a_max_horizon=6)

    def test_explicit_penalized_bootstrap_and_horizon_units(self):
        rng=np.random.default_rng(431)
        targets=np.array([[.7,.2,.1],[.6,.25,.15],[.65,.25,.1],[.6,.2,.2],[.68,.22,.1]])
        future=np.array(self.model['scenario_control']['real_returns'])[:,25:30]
        idx=rng.integers(0,len(future),(200,len(future)))
        counts=np.stack([np.bincount(row,minlength=len(future)) for row in idx])
        terminal,losses,_=hold_paths(targets,targets[0],.1,.23,future,.005)
        actual_log=bootstrap_log_risk(losses,counts)
        for i,row in enumerate(idx):
            np.testing.assert_allclose(actual_log[i],[expected_shortfall(-np.log1p(-x[row])) for x in losses],atol=1e-14)
        for h in [15,30]:
            raw,fees,_=hold_values(targets,targets[0],.1,.23,future,h,.005,self.settings)
            base=raw-np.maximum(0.,np.array([expected_shortfall(-np.log1p(-x)) for x in losses])-.3)/h
            bootstrap=[]
            for row in idx:
                q=hold_values(targets,targets[0],.1,.23,future[row],h,.005,self.settings)[0]
                risk=np.array([expected_shortfall(-np.log1p(-x[row])) for x in losses])
                bootstrap.append(q-np.maximum(0.,risk-.3)/h)
            bootstrap=np.array(bootstrap)
            expected=base-(bootstrap-bootstrap[:,0,None]).std(axis=0,ddof=1)
            actual,paid,_,diag=budget_values(targets,targets[0],.1,.23,future,h,.005,self.settings,counts,.3,True)
            np.testing.assert_allclose(actual,expected,atol=1e-14)
            np.testing.assert_array_equal(paid,fees)
            mean=budget_values(targets,targets[0],.1,.23,future,h,.005,self.settings,None,.3,False)[0]
            np.testing.assert_allclose(mean,base,atol=1e-14)

    def test_zero_coefficient_reproduces_previous_value(self):
        p=BudgetFusion('budget_uncertainty',self.model,self.params,self.config,30,self.settings,self.rule,'budget_uncertainty')
        targets=np.array([[.7,.2,.1],[.6,.25,.15],[.65,.25,.1],[.6,.2,.2],[.68,.22,.1]])
        args=(targets,targets[0],.1,.23,p.future,30,.005,self.settings,p.bootstrap_counts)
        actual=budget_values(*args,.3,True,coefficient=0.)[0]
        expected=uncertainty_values(*args)[0]
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        nonbinding=budget_values(*args,10.,True)[0]
        np.testing.assert_allclose(nonbinding,expected,atol=1e-14)

    def test_short_exact_causal_prefix_and_real_cost_gate(self):
        s=replace(self.scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in self.scenarios[0].observations))
        for variant in ['budget_uncertainty','budget_mean']:
            p=BudgetFusion(variant,self.model,self.params,self.config,5,self.settings,self.rule,variant)
            a=PlanAPolicy('plan_a_balanced',self.model,self.params,self.config,5)
            x=simulate(s,p,p.initial_target,SimulationConfig(5,100.,.005))
            y=simulate(s,a,a.initial_target,SimulationConfig(5,100.,.005))
            self.assertEqual(x.records,y.records)
            p=BudgetFusion(variant,self.model,self.params,self.config,30,self.settings,self.rule,variant)
            x=simulate(s,p,p.initial_target,SimulationConfig(30,100.,.005))
            changed=replace(s,observations=s.observations[:3]+tuple(replace(o,nominal_returns=(1.,-.2,.1)) for o in s.observations[3:]))
            q=BudgetFusion(variant,self.model,self.params,self.config,30,self.settings,self.rule,variant)
            y=simulate(changed,q,q.initial_target,SimulationConfig(30,100.,.005))
            for a,b in zip(x.records[:4],y.records[:4]): self.assertEqual(a.post_trade_holdings,b.post_trade_holdings)
            for r in x.records[1:]:
                if r.moved_amount>1e-8:
                    self.assertGreater(r.estimated_benefit,r.estimated_cost+.02)
                    self.assertLessEqual(.5*np.abs(np.array(r.pre_trade_holdings)/sum(r.pre_trade_holdings)-np.array(r.post_trade_holdings)/sum(r.post_trade_holdings)).sum(),.1+1e-10)
