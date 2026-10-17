"""Generated scenarios verify causal signals and scalar MPC path accounting."""
from dataclasses import replace
from itertools import product
import copy
import time
import unittest
import numpy as np
from safeinvest.contracts import Action, MarketObservation, PolicyContext, Scenario, SimulationConfig
from safebench.configuration import load_config
from safebench.engine import simulate, execute_trade
from safebench.metrics import expected_shortfall
from safebench.plan_a import PlanAPolicy
from safebench.plan_b import PlanBPolicy
from safebench.plan_c import PlanCPolicy, rollout, fit_control
from safebench.signals import fit_signals, forecast, causal_features
from safebench import ROOT


def fixture():
    rng = np.random.default_rng(772)
    scenarios = [Scenario(sid, tuple(MarketObservation(sid, t+1, tuple(row), .002)
                for t, row in enumerate(rng.uniform([-.3,-.08,0], [.4,.1,.03], (30,3))))) for sid in range(1,12)]
    config = load_config(ROOT/'configs/plan_b.json')
    config['plan_c'] = dict(enabled=True, evaluation_horizons=[5,30])
    entries = {f'balanced:{h}:1': dict(weights=[.5,.3,.2],review_threshold=.6,risk_budget=.5,validation_budget=.5) for h in (5,30)}
    model = dict(train_ids=list(range(1,12)), mu=[.05,.02,.01], cov=np.diag([.04,.01,.001]).tolist(),
        plan_a=dict(train_ids=list(range(1,12)),entries=entries),signals=fit_signals(scenarios))
    returns = np.asarray([[(1+np.asarray(o.nominal_returns))/(1+o.inflation)-1 for o in s.observations] for s in scenarios])
    model['scenario_control'] = dict(train_ids=model['train_ids'], real_returns=returns.tolist(),defensive={'5':[0,0,1],'30':[0,0,1]})
    return scenarios,config,model,dict(risk_ratio=1.,tracking_gamma=4.)


class SignalTests(unittest.TestCase):
    def test_causal_features_and_frozen_scaler(self):
        scenarios,config,model,params=fixture()
        original=copy.deepcopy(model['signals'])
        s=scenarios[0]
        for t in range(3,10):
            changed=replace(s,observations=s.observations[:t-1]+tuple(replace(o,nominal_returns=(2.,-0.9,.1)) for o in s.observations[t-1:]))
            np.testing.assert_array_equal(causal_features(s.observations[:t-1]),causal_features(changed.observations[:t-1]))
            self.assertEqual(forecast(s.observations[:t-1],model['signals']),forecast(changed.observations[:t-1],model['signals']))
        self.assertEqual(model['signals'],original)
        with self.assertRaises(PermissionError): fit_signals([Scenario(121,())])
        with self.assertRaises(PermissionError): fit_control([Scenario(121,())],config)

    def test_zero_narrow_exactly_reproduces_a_and_all_gates(self):
        scenarios,config,model,params=fixture()
        s=replace(scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in scenarios[0].observations))
        a=PlanAPolicy('plan_a_balanced',model,params,config,30)
        b=PlanBPolicy('plan_b_narrow_0',model,params,config,30)
        ra=simulate(s,a,a.initial_target,SimulationConfig())
        rb=simulate(s,b,b.initial_target,SimulationConfig())
        for x,y in zip(ra.records,rb.records):
            self.assertEqual(x.action,y.action); self.assertEqual(x.end_holdings,y.end_holdings)
            self.assertEqual(x.transaction_cost,y.transaction_cost)
        for name in ('plan_b_narrow_025','plan_b_wide_05','plan_b_period','plan_b_shifted'):
            p=PlanBPolicy(name,model,params,config,30)
            simulate(s,p,p.initial_target,SimulationConfig())

    def test_ood_and_short_history_fall_back(self):
        scenarios,config,model,params=fixture()
        p=PlanBPolicy('plan_b_wide_05',model,params,config,30)
        a=PlanAPolicy('plan_a_balanced',model,params,config,30)
        c=PolicyContext(2,scenarios[0].observations[:1],(80.,10.,10.),(.5,.3,.2),1.,100.,False,.0025)
        self.assertEqual(p.decide(c),a.decide(c))
        history=tuple(replace(o,nominal_returns=(100.,0.,0.)) for o in scenarios[0].observations[:3])
        c=replace(c,period=4,history=history)
        self.assertEqual(p.decide(c),a.decide(c)); self.assertTrue(p.last_diagnostics['fallback'])


class ControlTests(unittest.TestCase):
    def test_every_sequence_matches_independent_scalar_a_continuation(self):
        scenarios,config,model,params=fixture()
        for horizon in (5,30):
            p=PlanCPolicy('plan_c_l3',model,params,config,horizon)
            c=PolicyContext(2,(),(65.,20.,15.),(.5,.3,.2),1.,120.,False,.005,.23)
            for depth in (1,3):
                result=rollout(p,c,depth,1.)
                for index,seq in enumerate(product(range(4),repeat=depth)):
                    growth=[]; losses=[]
                    for path in p.future:
                        h=np.array(c.holdings); peak=c.real_peak; mdd=c.historical_max_drawdown; pending=False
                        target=c.target_weights
                        for step,t in enumerate(range(c.period-1,horizon)):
                            wealth=sum(h)
                            pending |= 1-wealth/peak >= p.review_threshold
                            ctx=PolicyContext(t+1,(),tuple(h),target,1.,peak,pending,c.transaction_cost_rate,mdd)
                            a=PlanAPolicy('plan_a_balanced',model,params,config,horizon)
                            d=a.decide(ctx) if not pending else None
                            if d is not None and d.action == Action.HUMAN_REVIEW: pending=True
                            kind=seq[step] if step<depth else 1
                            if pending or kind==0: w=h/wealth
                            elif kind==1: w=np.asarray(d.target_weights) if d.target_weights is not None else h/wealth
                            elif kind==2: w=p.anchor
                            else: w=p.defensive
                            trade=execute_trade(h,tuple(w),c.transaction_cost_rate)
                            h=np.asarray(trade.post_trade_holdings); mdd=max(mdd,1-sum(h)/peak); peak=max(peak,sum(h))
                            h*=1+path[t]; mdd=max(mdd,1-sum(h)/peak); peak=max(peak,sum(h)); target=tuple(w)
                        growth.append(np.log(sum(h)/sum(c.holdings))); losses.append(-np.log1p(-mdd))
                    self.assertAlmostEqual(result['growth'][index],np.mean(growth),places=12)
                    self.assertAlmostEqual(result['risk'][index],expected_shortfall(losses,.9),places=12)

    def test_c_policy_cannot_observe_mutated_future(self):
        scenarios,config,model,params=fixture()
        s=replace(scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in scenarios[0].observations))
        changed=replace(s,observations=s.observations[:3]+tuple(replace(o,nominal_returns=(.8,-.5,.01),inflation=.1) for o in s.observations[3:]))
        first=PlanCPolicy('plan_c_l3',model,params,config,5)
        second=PlanCPolicy('plan_c_l3',model,params,config,5)
        a=simulate(s,first,first.initial_target,SimulationConfig(5))
        b=simulate(changed,second,second.initial_target,SimulationConfig(5))
        self.assertEqual(a.records[:3],b.records[:3])
        self.assertEqual(a.records[3].action,b.records[3].action)
        self.assertEqual(a.records[3].target_weights,b.records[3].target_weights)
        self.assertEqual(a.records[3].policy_diagnostics,b.records[3].policy_diagnostics)

    def test_c_screening_does_not_shorten_a_evaluation(self):
        from safebench.experiment import evaluate
        scenarios,config,model,params=fixture()
        config['horizons']=[5,30]; config['plan_c']['evaluation_horizons']=[30]
        s=replace(scenarios[0],scenario_id=100,observations=tuple(replace(o,scenario_id=100) for o in scenarios[0].observations))
        m,_=evaluate([s],model,{'plan_a_balanced':params,'plan_c_l1':params},config,names=['plan_a_balanced','plan_c_l1'])
        self.assertEqual(set(m[m.policy=='plan_a_balanced'].horizon),{5,30})
        self.assertEqual(set(m[m.policy=='plan_c_l1'].horizon),{30})

    def test_invalid_a_value_enters_review_in_actual_and_simulated_state(self):
        _,config,model,params=fixture()
        model['cov']=np.diag([100.,100.,100.]).tolist()
        p=PlanCPolicy('plan_c_l3',model,params,config,5)
        c=PolicyContext(2,(),(90.,5.,5.),(.5,.3,.2),1.,100.,False,.005,0.)
        self.assertEqual(p.decide(c).action,Action.HUMAN_REVIEW)
        c=replace(c,holdings=(50.,30.,20.))
        p.future[:,1,:]=[4.,0.,0.]
        r=rollout(p,c,1,1.)
        # Hold in year 2; the resulting drift makes A invalid in year 3.
        terminal=np.asarray(c.holdings)[None,:]*np.prod(1+p.future[:,1:5,:],axis=1)
        np.testing.assert_allclose(r['growth'][0],np.log(terminal.sum(axis=1)/100.).mean(),atol=1e-12)

    def test_historic_drawdown_memory_and_truncation(self):
        _,config,model,params=fixture()
        p=PlanCPolicy('plan_c_l3',model,params,config,5)
        c=PolicyContext(5,(),(50.,30.,20.),(.5,.3,.2),1.,100.,False,.0025,0.)
        r=rollout(p,c,3,1.)
        self.assertEqual(len(r['sequences']),4)
        severe=rollout(p,replace(c,historical_max_drawdown=.8),3,1.)
        self.assertTrue((severe['risk']>=-np.log(.2)-1e-12).all())
        self.assertTrue((severe['objective']<r['objective']).all())

    def test_fee_gate_and_hold_first_reference(self):
        _,config,model,params=fixture()
        p=PlanCPolicy('plan_c_l3',model,params,config,5)
        c=PolicyContext(2,(),(80.,10.,10.),(.5,.3,.2),1.,100.,False,.005,.1)
        result=rollout(p,c,3,1.)
        decision=p.decide(c); d=p.last_diagnostics
        self.assertAlmostEqual(d['hold_objective'],max(result['objective'][result['sequences'][:,0]==0]))
        excess=(d['benefit']-d['first_cost'])/(1-d['first_cost'])
        self.assertAlmostEqual(excess,-np.expm1(d['hold_objective']-d['best_objective']))
        if decision.target_weights is not None:
            actual=execute_trade(c.holdings,decision.target_weights,c.transaction_cost_rate).transaction_cost/sum(c.holdings)
            self.assertAlmostEqual(actual,decision.estimated_cost)
            self.assertGreater(decision.estimated_benefit,actual+p.buffer)

    def test_engine_supplies_historic_fee_and_recovery_memory(self):
        contexts=[]
        class Spy:
            name='spy'
            def decide(self,c):
                contexts.append(c)
                from safeinvest.contracts import Decision
                return Decision(Action.KEEP,('HOLD',),'Hold.')
        s=Scenario(100,tuple(MarketObservation(100,t+1,(r,r,r),0.) for t,r in enumerate((-.4,1.,0.))))
        simulate(s,Spy(),(.5,.3,.2),SimulationConfig(3,100.,.01))
        self.assertGreater(contexts[1].historical_max_drawdown,.4)
        self.assertEqual(contexts[1].real_peak,sum(contexts[1].holdings))

    def test_identical_asset_returns_and_expensive_trades_hold(self):
        _,config,model,params=fixture()
        model['scenario_control']['real_returns']=np.full((11,30,3),.02).tolist()
        p=PlanCPolicy('plan_c_l3_no_risk',model,params,config,5)
        c=PolicyContext(2,(),(80.,10.,10.),(.5,.3,.2),1.,100.,False,.1,0.)
        self.assertEqual(p.decide(c).action,Action.KEEP)
