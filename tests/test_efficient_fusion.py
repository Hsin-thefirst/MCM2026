"""Generated checks for stronger net-value gates and reversal state."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from efficient_fusion import EfficientFusion
from fusion_policy import FusionPolicy
from run_fusion_research import scalar_check
from test_plan_bc import fixture
from safeinvest.contracts import SimulationConfig,PolicyContext
from safebench.engine import simulate


class EfficientTests(unittest.TestCase):
    def setUp(self):
        self.scenarios,self.config,self.model,self.params=fixture()
        self.base=json.loads((Path(__file__).resolve().parents[1]/'configs/fusion_research.json').read_text())
        self.rules=json.loads((Path(__file__).resolve().parents[1]/'configs/fusion_efficiency.json').read_text())['rules']
        self.s=replace(self.scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in self.scenarios[0].observations))

    def test_base_exact_and_causal_prefix(self):
        for h in (5,30):
            a=FusionPolicy('fusion_drawdown',self.model,self.params,self.config,h,self.base)
            b=EfficientFusion('efficient_base',self.model,self.params,self.config,h,self.base,self.rules['efficient_base'])
            ra=simulate(self.s,a,a.initial_target,SimulationConfig(h,100.,.005))
            rb=simulate(self.s,b,b.initial_target,SimulationConfig(h,100.,.005))
            for x,y in zip(ra.records,rb.records):
                self.assertEqual(x.action,y.action);self.assertEqual(x.post_trade_holdings,y.post_trade_holdings)
            changed=replace(self.s,observations=self.s.observations[:3]+tuple(replace(o,nominal_returns=(1.,-.2,.1)) for o in self.s.observations[3:]))
            c=EfficientFusion('efficient_base',self.model,self.params,self.config,h,self.base,self.rules['efficient_base'])
            rc=simulate(changed,c,c.initial_target,SimulationConfig(h,100.,.005))
            for x,y in zip(rb.records[:4],rc.records[:4]):self.assertEqual(x.post_trade_holdings,y.post_trade_holdings)

    def test_stronger_gates_and_accounting(self):
        for name,rule in self.rules.items():
            p=EfficientFusion(name,self.model,self.params,self.config,30,self.base,rule)
            result=simulate(self.s,p,p.initial_target,SimulationConfig(30,100.,.005))
            scalar_check(result,self.s)
            for r in result.records[1:]:
                if r.moved_amount>1e-8:
                    self.assertGreater(r.estimated_benefit,r.estimated_cost+rule['buffer'])
                    self.assertFalse(r.policy_diagnostics['reversal_blocked'][r.policy_diagnostics['chosen']])

    def test_reversal_blocks_only_next_opposite_direction(self):
        p=EfficientFusion('efficient_reversal',self.model,self.params,self.config,30,self.base,self.rules['efficient_reversal'])
        c=PolicyContext(3,self.scenarios[0].observations[:2],(80.,10.,10.),(.5,.3,.2),1.,100.,False,.0025)
        p.last_trade_period=2;p.last_equity_direction=1
        p.decide(c);d=p.last_diagnostics
        targets=np.array(d['targets']);blocked=np.array(d['reversal_blocked'])
        np.testing.assert_array_equal(blocked,targets[:,0]<.8)
        p.last_trade_period=1;p.last_equity_direction=1
        p.decide(c);self.assertFalse(any(p.last_diagnostics['reversal_blocked']))
