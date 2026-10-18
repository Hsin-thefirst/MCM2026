"""Scalar continuation and explicit-resample checks for fixed value extensions."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fusion_value_extensions import (ValueFusion,continuation_values,weighted_bootstrap_risk,
                                     hold_paths,uncertainty_values)
from fusion_policy import hold_values,tail
from safeinvest.contracts import Action,PolicyContext,SimulationConfig
from safebench.engine import simulate
from safebench.metrics import expected_shortfall
from safebench.plan_a import PlanAPolicy
from test_plan_bc import fixture


class FusionValueTests(unittest.TestCase):
    def setUp(self):
        self.scenarios,self.config,self.model,self.params = fixture()
        self.base = json.loads((ROOT/'configs/fusion_research.json').read_text())
        self.rule = dict(buffer=.02,short_a_max_horizon=6)

    def test_weighted_bootstrap_matches_explicit_empirical_samples(self):
        rng = np.random.default_rng(431)
        for n in [7,11,64,96]:
            losses = rng.choice([.1,.2,.3,.8],(5,n))
            idx = rng.integers(0,n,(200,n))
            counts = np.stack([np.bincount(row,minlength=n) for row in idx])
            es,p90 = weighted_bootstrap_risk(losses,counts)
            for i,row in enumerate(idx):
                explicit = losses[:,row]
                np.testing.assert_allclose(es[i],[expected_shortfall(x) for x in explicit],atol=1e-14)
                np.testing.assert_allclose(p90[i],np.quantile(explicit,.9,axis=1),atol=1e-14)

    def test_uncertainty_values_match_full_resample_objectives(self):
        p = ValueFusion('uncertainty',self.model,self.params,self.config,30,self.base,self.rule,'uncertainty')
        targets = np.array([[.7,.2,.1],[.6,.25,.15],[.65,.25,.1],[.6,.2,.2],[.68,.22,.1]])
        future = p.future[:,27:30]
        q,fees,_ = hold_values(targets,targets[0],.1,.23,future,30,.005,self.base)
        terminal,losses,_ = hold_paths(targets,targets[0],.1,.23,future,.005)
        rng = np.random.default_rng(20261021)
        indices = rng.integers(0,len(future),(200,len(future)))
        boots = np.array([hold_values(targets,targets[0],.1,.23,future[idx],30,.005,self.base)[0] for idx in indices])
        expected = q-(boots-boots[:,0,None]).std(axis=0,ddof=1)
        actual,cost,_,diag = uncertainty_values(targets,targets[0],.1,.23,future,30,.005,self.base,p.bootstrap_counts)
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        np.testing.assert_array_equal(fees,cost)
        self.assertEqual(diag['gain_sd'][0],0.)

    def test_a_continuation_matches_scalar_fee_peak_and_review(self):
        cases=[(5,3,100.,1.,.23,.12),(30,2,100.,1.,.23,.12),
               (30,29,100.,1.,.23,.12),(30,2,120.,1.1,.30,.6)]
        for horizon,period,real_peak,inflation,history,threshold in cases:
            self.model['plan_a']['entries'][f'balanced:{horizon}:1']['review_threshold'] = threshold
            p = ValueFusion('continuation',self.model,self.params,self.config,horizon,self.base,self.rule,'a_continuation')
            c = PolicyContext(period,(),(65.,20.,15.),(.5,.3,.2),inflation,real_peak,False,.005,history)
            targets = p.candidates(c,p.adviser._decide(c))
            actual,fees,_,diag = continuation_values(p,c,targets)
            values=[];costs=[];risk=[];p90=[];reviewed=0
            for target in targets:
                final=[];losses=[];paid=[]
                for future in p.future:
                    holdings=np.array(c.holdings)/sum(c.holdings)
                    peak=c.real_peak/(sum(c.holdings)/c.cumulative_inflation)
                    maximum=history;pending=False;paid_path=0.;current_target=c.target_weights
                    for step,t in enumerate(range(period-1,horizon)):
                        wealth=sum(holdings)
                        if step==0: chosen=target
                        elif pending: chosen=holdings/wealth
                        else:
                            context=PolicyContext(t+1,(),tuple(holdings),tuple(current_target),1.,peak,False,.005,maximum)
                            advice=p.adviser._decide(context)
                            if advice.action==Action.HUMAN_REVIEW: pending=True;reviewed+=1
                            chosen=np.array(advice.target_weights) if advice.target_weights is not None else holdings/wealth
                        fee=.005*sum(abs(wealth*x-v) for x,v in zip(chosen,holdings))/2
                        paid_path+=fee
                        post=wealth-fee
                        maximum=max(maximum,1-post/peak);peak=max(peak,post)
                        holdings=post*chosen*(1+future[t])
                        wealth=sum(holdings)
                        maximum=max(maximum,1-wealth/peak);peak=max(peak,wealth)
                        current_target=chosen
                    final.append(wealth);losses.append(maximum);paid.append(paid_path)
                values.append(np.log(final).mean()/horizon)
                risk.append(expected_shortfall(losses));p90.append(np.quantile(losses,.9));costs.append(np.mean(paid))
            expected=np.array(values)-.03*(np.array(risk)-history)-.5*np.maximum(0.,np.array(p90)-p90[1]-.01)
            np.testing.assert_allclose(actual,expected,atol=1e-14)
            np.testing.assert_allclose(diag['mean_continuation_fees'],costs,atol=1e-14)
            np.testing.assert_allclose(fees,.005*.5*np.abs(targets-np.array([.65,.2,.15])).sum(axis=1))
            if horizon==30 and period==2 and threshold==.12: self.assertGreater(reviewed,0)

    def test_short_exactness_and_causal_trade_gates(self):
        s=replace(self.scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in self.scenarios[0].observations))
        for variant in ['a_continuation','uncertainty']:
            p=ValueFusion(variant,self.model,self.params,self.config,5,self.base,self.rule,variant)
            a=PlanAPolicy('plan_a_balanced',self.model,self.params,self.config,5)
            x=simulate(s,p,p.initial_target,SimulationConfig(5,100.,.005))
            y=simulate(s,a,a.initial_target,SimulationConfig(5,100.,.005))
            self.assertEqual(x.records,y.records)
            p=ValueFusion(variant,self.model,self.params,self.config,30,self.base,self.rule,variant)
            x=simulate(s,p,p.initial_target,SimulationConfig(30,100.,.005))
            changed=replace(s,observations=s.observations[:3]+tuple(replace(o,nominal_returns=(1.,-.2,.1)) for o in s.observations[3:]))
            q=ValueFusion(variant,self.model,self.params,self.config,30,self.base,self.rule,variant)
            y=simulate(changed,q,q.initial_target,SimulationConfig(30,100.,.005))
            for a,b in zip(x.records[:4],y.records[:4]): self.assertEqual(a.post_trade_holdings,b.post_trade_holdings)
            for r in x.records[1:]:
                if r.moved_amount>1e-8:
                    self.assertGreater(r.estimated_benefit,r.estimated_cost+.02)
                    self.assertLessEqual(.5*np.abs(np.array(r.pre_trade_holdings)/sum(r.pre_trade_holdings)-np.array(r.post_trade_holdings)/sum(r.post_trade_holdings)).sum(),.1+1e-10)
