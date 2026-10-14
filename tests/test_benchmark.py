import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safebench import ROOT
from safeinvest.contracts import Scenario, MarketObservation, SimulationConfig, Decision, Action
from safebench.engine import execute_trade, simulate
from safebench.metrics import compute_metrics, expected_shortfall, binomial_upper, summarize, paired_comparisons
from safebench.data import load_partition, folds
from safebench.policies import StaticPolicy, GatedPolicy, fit_model, make_policy, POLICY_NAMES, EQUAL
from safebench.experiment import claim_final_lock, verify_bundle, digest, code_digest, versions, evaluate

CONFIG=json.loads((ROOT/'configs/benchmark.json').read_text())

def scenario(rows,sid=1):
    return Scenario(sid,tuple(MarketObservation(sid,t+1,tuple(row[:3]),row[3]) for t,row in enumerate(rows)))

class AccountingTests(unittest.TestCase):
    def test_transfer_once(self):
        t=execute_trade((0,0,100),(.2,0,.8),.0025)
        self.assertAlmostEqual(t.moved_amount,20); self.assertAlmostEqual(t.transaction_cost,.05)
        np.testing.assert_allclose(t.post_trade_holdings,(19.99,0,79.96))
    def test_equal_initial_fee(self):
        t=execute_trade((0,0,100),EQUAL,.0025)
        self.assertAlmostEqual(t.transaction_cost,1/6)
        self.assertAlmostEqual(sum(t.post_trade_holdings),100-1/6)
    def test_no_turnover(self):
        t=execute_trade((20,30,50),(.2,.3,.5),.01)
        self.assertEqual(t.transaction_cost,0)
    def test_invalid_trade(self):
        for weights in [(-.1,.5,.6),(.2,.2,.2),(float('nan'),0,1)]:
            with self.assertRaises(ValueError): execute_trade((0,0,100),weights,.0025)
        with self.assertRaises(ValueError): execute_trade((0,0,0),EQUAL,.0025)
    def test_manual_compounding_inflation(self):
        s=scenario([( .1,0,0,.05),(-.2,0,0,.02)])
        p=StaticPolicy('hold',(1.,0,0),'hold')
        r=simulate(s,p,p.weights,SimulationConfig(2))
        wealth=(100-.25)*1.1*.8/1.05/1.02
        m=compute_metrics(r)
        self.assertAlmostEqual(m['terminal_real_wealth'],wealth)
        self.assertAlmostEqual(m['real_cagr'],np.sqrt(wealth/100)-1)
        self.assertAlmostEqual(m['real_max_drawdown'],1-.8/1.02)
        self.assertAlmostEqual(m['total_nominal_cost'],.25)
        self.assertEqual(m['trade_rate'],0)
    def test_buy_hold_drift(self):
        p=StaticPolicy('hold',(.5,0,.5),'hold')
        r=simulate(scenario([(1,0,0,0),(0,0,0,0)]),p,p.weights,SimulationConfig(2,100,0))
        self.assertEqual(r.records[1].pre_trade_holdings,(100,0,50))
        self.assertEqual(r.records[1].post_trade_holdings,(100,0,50))
    def test_rebalance_manual_second_fee(self):
        p=StaticPolicy('annual',(.5,0,.5))
        r=simulate(scenario([(1,0,0,0),(0,0,0,0)]),p,p.weights,SimulationConfig(2,100,.01))
        self.assertAlmostEqual(r.records[0].transaction_cost,.5)
        self.assertAlmostEqual(r.records[1].moved_amount,24.875)
        self.assertAlmostEqual(r.records[1].real_wealth,149.25-.24875)
    def test_fee_only_drawdown(self):
        p=StaticPolicy('annual',(1.,0,0))
        r=simulate(scenario([(0,0,0,0)]),p,p.weights,SimulationConfig(1,100,.01))
        self.assertAlmostEqual(compute_metrics(r)['real_max_drawdown'],.01)
    def test_fee_time_drawdown_survives_recovery(self):
        p=StaticPolicy('annual',(.5,0,.5))
        r=simulate(scenario([(1,0,0,0),(.1,.1,.1,0)]),p,p.weights,SimulationConfig(2,100,.01))
        self.assertAlmostEqual(compute_metrics(r)['real_max_drawdown'],.005)
    def test_conservation_random(self):
        rng=np.random.default_rng(91)
        for _ in range(100):
            h=rng.uniform(0,200,3); w=rng.dirichlet(np.ones(3)); t=execute_trade(h,tuple(w),.0025)
            self.assertAlmostEqual(sum(t.post_trade_holdings)+t.transaction_cost,sum(h))
            self.assertTrue(0<=t.moved_amount<=sum(h)+1e-9)
    def test_missing_horizon(self):
        p=StaticPolicy('x',EQUAL)
        with self.assertRaises(ValueError): simulate(scenario([(0,0,0,0)]),p,EQUAL,SimulationConfig(2))

class CausalityTests(unittest.TestCase):
    def test_causal_prefix(self):
        seen=[]
        class Spy:
            name='spy'
            def decide(self,c):
                seen.append(c)
                self_test.assertEqual(len(c.history),c.period-1)
                return Decision(Action.KEEP,('HOLD',),'hold')
        self_test=self
        s=scenario([(.1,0,0,0),(.2,0,0,.01),(.3,0,0,0)])
        a=simulate(s,Spy(),EQUAL,SimulationConfig(3))
        first_seen=list(seen); seen.clear()
        b=simulate(replace(s,observations=s.observations[:2]+(MarketObservation(1,3,(-.9,2,1),.2),)),Spy(),EQUAL,SimulationConfig(3))
        self.assertEqual(first_seen,seen)
        self.assertEqual(a.records[:2],b.records[:2])
    def test_review_absorbing(self):
        class Review:
            name='review'
            def decide(self,c):
                if c.period>2: raise AssertionError('Policy called while review pending')
                return Decision(Action.HUMAN_REVIEW,('ALERT',),'review')
        r=simulate(scenario([(0,0,0,0)]*4),Review(),EQUAL,SimulationConfig(4))
        self.assertEqual(r.records[1].action,Action.HUMAN_REVIEW)
        self.assertEqual(r.records[2].reason_codes,('REVIEW_PENDING',))
        self.assertEqual(sum(x.transaction_cost for x in r.records[1:]),0)
    def test_gate_rejects_unsupported_trade(self):
        class Bad:
            name='bad'; requires_benefit_gate=True; buffer=0
            def decide(self,c):
                return Decision(Action.CHANGE_TARGET,('TRADE',),'x',(1.,0,0),0.,0.)
        with self.assertRaises(ValueError): simulate(scenario([(0,0,0,0)]*2),Bad(),EQUAL,SimulationConfig(2))
    def test_rebalance_cannot_change_target(self):
        class Bad:
            name='bad'
            def decide(self,c): return Decision(Action.REBALANCE,('TRADE',),'x',(1.,0,0))
        with self.assertRaises(ValueError): simulate(scenario([(0,0,0,0)]*2),Bad(),EQUAL,SimulationConfig(2))
    def test_fit_final_refused(self):
        with self.assertRaises(PermissionError): fit_model([scenario([(0,0,0,0)]*30,121)],CONFIG)

class EvaluationTests(unittest.TestCase):
    def test_es_fractional_mass(self):
        self.assertAlmostEqual(expected_shortfall([0,1,2],.5),5/3)
    def test_binomial_zero_events_uncertainty(self):
        self.assertAlmostEqual(binomial_upper(0,60),1-.05**(1/60))
        self.assertEqual(binomial_upper(60,60),1)
    def test_fold_isolation(self):
        groups=folds(range(1,121),5,42)
        self.assertEqual(sorted(sum(groups,[])),list(range(1,121)))
        self.assertEqual(groups,folds(range(1,121),5,42))
        self.assertTrue(all(len(g)==24 for g in groups))
    def test_partition_poison_holdout(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            path=Path(td)/'data.csv'
            with path.open('w') as f:
                f.write('scenario_id,period,equity_return,bond_return,bill_proxy_return,inflation\n')
                for sid in range(1,121):
                    for t in range(1,31): f.write(f'{sid},{t},0,0,0,0\n')
                f.write('HOLDOUT IS DELIBERATELY INVALID AND MUST NOT BE PARSED\n')
            ss,_=load_partition(path); self.assertEqual(len(ss),120)
            with self.assertRaises(PermissionError): load_partition(path,'final')
    def test_final_guard_exclusive(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            p=Path(td)/'lock'; claim_final_lock(p,'abc')
            with self.assertRaises(FileExistsError): claim_final_lock(p,'abc')
    def test_frozen_tamper_detection(self):
        b=dict(code_sha256=code_digest(),dependencies=versions(),final_scenarios=list(range(121,181)))
        b['bundle_sha256']=digest(b); verify_bundle(b)
        b['final_scenarios']=[121]
        with self.assertRaises(ValueError): verify_bundle(b)
    def test_paired_identical_zero_interval(self):
        import pandas as pd
        rows=[]
        for name in ['equal_annual','copy']:
            for sid in range(1,11): rows.append(dict(policy=name,scenario_id=sid,horizon=30,cost_rate=.0025,real_cagr=sid*.001,real_max_drawdown=sid*.01))
        result=paired_comparisons(pd.DataFrame(rows),bootstrap=100)
        self.assertTrue((result[['difference','lower95','upper95']]==0).all().all())
    def test_unpaired_rejected(self):
        import pandas as pd
        rows=[dict(policy=n,scenario_id=s,horizon=30,cost_rate=.0025,real_cagr=.1,real_max_drawdown=.2) for n,s in [('equal_annual',1),('x',2)]]
        with self.assertRaises(ValueError): paired_comparisons(pd.DataFrame(rows),bootstrap=10)
    def test_all_policies_smoke_and_gate(self):
        ss,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
        model=fit_model(ss[:5],CONFIG)
        for name in POLICY_NAMES:
            p=make_policy(name,model,CONFIG['tuning_grid'][0],CONFIG)
            r=simulate(ss[5],p,p.initial_target,SimulationConfig(30))
            self.assertEqual(len(r.records),30)
            self.assertGreater(compute_metrics(r)['terminal_real_wealth'],0)
            if p.requires_benefit_gate:
                for x in r.records[1:]:
                    if x.action in (Action.REBALANCE,Action.CHANGE_TARGET): self.assertGreater(x.estimated_benefit,x.estimated_cost+p.buffer)
    def test_evaluation_train_overlap_rejected(self):
        ss,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
        model=fit_model(ss[:5],CONFIG)
        with self.assertRaises(ValueError): evaluate(ss[:1],model,{},CONFIG,names=['equal_annual'])


class FinalWorkflowTests(unittest.TestCase):
    def test_freeze_and_final_on_generated_fixture_only(self):
        """Exercise final machinery on generated zero-return data, never the supplied holdout."""
        from unittest.mock import patch
        import pandas as pd
        from safebench.experiment import freeze, run_final, manifest
        with tempfile.TemporaryDirectory(dir=ROOT/'benchmark') as td:
            root=Path(td); data=root/'synthetic.csv'; dev=root/'dev'; dev.mkdir()
            with data.open('w') as f:
                f.write('scenario_id,period,equity_return,bond_return,bill_proxy_return,inflation\n')
                for sid in range(1,181):
                    for t in range(1,31): f.write(f'{sid},{t},0,0,0,0\n')
            config=dict(CONFIG,inner_folds=2,tuning_grid=[CONFIG['tuning_grid'][0]],bootstrap=50,sensitivity_costs=[.005])
            _,sha=load_partition(data)
            (dev/'manifest.json').write_text(json.dumps(manifest(config,sha,list(range(1,121)),'development')))
            pd.DataFrame([dict(profile_id='A',horizon=5,loss_tolerance=.1,liquidity='high',portfolio='human_review',explanation='fixture')]).to_csv(dev/'suitability.csv',index=False)
            bundle=root/'frozen.json'
            freeze(config,data,'balanced_hold',bundle,dev)
            frozen_code_hash=code_digest()
            with patch('safebench.experiment.BASE',root),patch('safebench.experiment.code_digest',return_value=frozen_code_hash),patch('safebench.report.generate_report'):
                with self.assertRaises(PermissionError): run_final(bundle,data,root/'no_ack',False)
                self.assertFalse((root/'FINAL_EVALUATION_LOCK.json').exists())
                run_final(bundle,data,root/'final',True)
                with self.assertRaises(FileExistsError): run_final(bundle,data,root/'second_final',True)
            result=pd.read_csv(root/'final/scenario_metrics.csv')
            self.assertEqual(set(result.scenario_id),set(range(121,181)))
            self.assertTrue((result.real_cagr<=0).all())
            self.assertTrue(json.loads((root/'final/validation.json').read_text())['all_checks_passed'])
    def test_cvar_infeasible_fails_closed(self):
        config=dict(CONFIG,cvar_limit=-.5)
        with self.assertRaisesRegex(RuntimeError,'infeasible'):
            fit_model([scenario([(0,0,0,0)]*30)],config)

if __name__=='__main__': unittest.main()
