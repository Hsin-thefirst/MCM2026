"""Small, complementary comparators; deployment policies explicitly enforce a cost gate."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from safeinvest.contracts import Action, Decision
from .data import real_rows

PORTFOLIOS={'conservative':(.2,.3,.5),'balanced':(.5,.35,.15),'growth':(.75,.2,.05)}
EQUAL=(1/3,1/3,1/3)
BALANCED_BOUNDS=((.2,.65),(.1,.7),(.1,.6))

def grid(bounds=BALANCED_BOUNDS,step=.05):
    n=round(1/step)
    return np.array([(i/n,j/n,(n-i-j)/n) for i in range(n+1) for j in range(n-i+1)
        if all(lo-1e-10<=w<=hi+1e-10 for w,(lo,hi) in zip((i/n,j/n,(n-i-j)/n),bounds))])

def moments(rows, shrink=.25):
    cov=np.cov(rows,rowvar=False,ddof=1)
    return rows.mean(axis=0),(1-shrink)*cov+shrink*np.diag(np.diag(cov))+np.eye(3)*1e-10

def optimize(mu,cov,gamma=5.,minimum_variance=False):
    bounds=((0.,1.),)*3 if minimum_variance else BALANCED_BOUNDS
    objective=(lambda w: w@cov@w) if minimum_variance else (lambda w: -mu@w+.5*gamma*(w@cov@w))
    fit=minimize(objective,EQUAL,method='SLSQP',bounds=bounds,
                 constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                 options={'ftol':1e-12,'maxiter':200})
    if not fit.success or abs(fit.x.sum()-1)>1e-7:
        raise RuntimeError('Optimizer failed: '+fit.message)
    w=np.clip(fit.x,0,1)
    return tuple(w/w.sum())

def fit_model(scenarios,config):
    if not scenarios or any(s.scenario_id>120 for s in scenarios):
        raise PermissionError('Fit accepts development scenarios only')
    rows=real_rows(scenarios)
    mu,cov=moments(rows,config['covariance_shrinkage'])
    # Tail-risk comparator uses the official simplex: an arbitrary equity floor
    # must not silently override its risk constraint.
    weights=grid(bounds=((0.,1.),)*3)
    losses=-(rows@weights.T)
    # Equal-probability annual samples; one-year CVaR is a proxy, never a path MDD guarantee.
    from .metrics import expected_shortfall
    tails=np.array([expected_shortfall(losses[:,i],config['cvar_alpha']) for i in range(len(weights))])
    feasible=tails<=config['cvar_limit']
    if not feasible.any():
        raise RuntimeError('CVaR cap infeasible on declared grid; no silent fallback')
    index=int(np.argmax(np.where(feasible,weights@mu,-np.inf)))
    inv=1/np.sqrt(np.diag(cov)); inv/=inv.sum()
    model = dict(train_ids=[s.scenario_id for s in scenarios],mu=mu.tolist(),cov=cov.tolist(),
                minvar=optimize(mu,cov,minimum_variance=True),inverse_vol=tuple(inv),
                meanvar=optimize(mu,cov,gamma=config['fixed_gamma']),cvar=tuple(weights[index]),
                cvar_feasible=bool(feasible.any()),cvar_estimate=float(tails[index]))
    from .plan_a import enabled, fit_plan_a
    if enabled(config):
        model['plan_a'] = fit_plan_a(scenarios, config)
    from . import plan_b, plan_c
    if plan_b.enabled(config):
        from .signals import fit_signals
        model['signals'] = fit_signals(scenarios)
    if plan_c.enabled(config): model['scenario_control'] = plan_c.fit_control(scenarios, config)
    return model

@dataclass
class StaticPolicy:
    name: str
    weights: tuple
    mode: str='annual'
    interval: int=3
    threshold: float=.05
    requires_benefit_gate: bool=False

    @property
    def initial_target(self): return self.weights

    def decide(self,c):
        current=np.array(c.holdings)/sum(c.holdings)
        trade=self.mode=='annual' or (self.mode=='periodic' and (c.period-1)%self.interval==0) or (
            self.mode=='threshold' and np.max(np.abs(current-self.weights))>self.threshold)
        if trade:
            return Decision(Action.REBALANCE,('COMPARATOR_REBALANCE',),'Restore the fixed target allocation.',self.weights)
        return Decision(Action.KEEP,('COMPARATOR_HOLD',),'Hold the existing assets.')

class GatedPolicy:
    requires_benefit_gate=True
    def __init__(self,name,model,params,config,adaptive=False,fixed=False,review=False):
        self.name=name
        self.mu=np.array(model['mu'])
        self.cov=np.array(model['cov'])
        self.gamma=params['gamma']
        self.buffer=params['buffer']
        self.window=params.get('window',5)
        self.prior_strength=config['prior_strength']
        self.adaptive,self.fixed,self.review=adaptive,fixed,review
        self.review_threshold=config['review_drawdown']
        self.candidates=grid()
        self.initial_target=PORTFOLIOS['balanced']
    def decide(self,c):
        if self.review and 1-sum(c.holdings)/c.cumulative_inflation/c.real_peak >= self.review_threshold:
            return Decision(Action.HUMAN_REVIEW,('DRAWDOWN_ALERT',),
                'Your purchasing-power loss has crossed our review threshold. A human adviser should reassess your plan. We will hold your existing investments while review is pending.')
        mu=self.mu.copy()
        if self.adaptive and c.history:
            recent=c.history[-self.window:]
            rows=np.array([[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns] for o in recent])
            mu=(self.prior_strength*mu+rows.sum(axis=0))/(self.prior_strength+len(rows))
        current=np.array(c.holdings)/sum(c.holdings)
        weights=np.array([PORTFOLIOS['balanced']]) if self.fixed else self.candidates
        utility=1+weights@mu-.5*self.gamma*np.einsum('ij,jk,ik->i',weights,self.cov,weights)
        costs=c.transaction_cost_rate*.5*np.abs(weights-current).sum(axis=1)
        index=int(np.argmax((1-costs)*utility))
        target=tuple(weights[index]); u=float(utility[index])
        current_u=1+mu@current-.5*self.gamma*(current@self.cov@current)
        if u<=0 or not np.isfinite(u):
            return Decision(Action.HUMAN_REVIEW,('INVALID_ESTIMATE',),'The return estimate is unreliable. Please seek human review.')
        benefit=float((u-current_u)/u)
        cost=float(costs[index])
        if benefit>cost+self.buffer:
            action=Action.REBALANCE if np.allclose(target,c.target_weights,rtol=0,atol=1e-10) else Action.CHANGE_TARGET
            return Decision(action,('BENEFIT_EXCEEDS_COST',),
                'This allocation has a higher estimated risk-adjusted value after trading costs. The estimate is uncertain; losses remain possible.',
                target,benefit,cost)
        return Decision(Action.KEEP,('BENEFIT_BELOW_COST',),'The estimated improvement does not justify trading costs and the safety margin.',
                        estimated_benefit=benefit,estimated_cost=cost)

POLICY_NAMES=['equal_annual','equal_hold','bills_hold','conservative_annual','balanced_annual','growth_annual',
              'balanced_hold','balanced_periodic','balanced_threshold','minimum_variance','inverse_volatility',
              'mean_variance','cvar_constrained','balanced_gate','cost_aware','adaptive','adaptive_review']

def policy_names(config):
    from .plan_a import enabled, POLICIES
    from . import plan_b, plan_c
    return POLICY_NAMES + (POLICIES if enabled(config) else []) + (plan_b.POLICIES if plan_b.enabled(config) else []) + (plan_c.POLICIES if plan_c.enabled(config) else [])


def make_policy(name,model,params,config,horizon=None):
    from .plan_a import POLICIES, PlanAPolicy
    if name in POLICIES:
        return PlanAPolicy(name, model, params, config, horizon)
    from . import plan_b, plan_c
    if name in plan_b.POLICIES: return plan_b.PlanBPolicy(name, model, params, config, horizon)
    if name in plan_c.POLICIES: return plan_c.PlanCPolicy(name, model, params, config, horizon)
    fixed={'equal_annual':(EQUAL,'annual'),'equal_hold':(EQUAL,'hold'),'bills_hold':((0.,0.,1.),'hold'),
           **{p+'_annual':(w,'annual') for p,w in PORTFOLIOS.items()},
           'balanced_hold':(PORTFOLIOS['balanced'],'hold'),'balanced_periodic':(PORTFOLIOS['balanced'],'periodic'),
           'balanced_threshold':(PORTFOLIOS['balanced'],'threshold'),
           'minimum_variance':(tuple(model['minvar']),'annual'),'inverse_volatility':(tuple(model['inverse_vol']),'annual'),
           'mean_variance':(tuple(model['meanvar']),'annual'),'cvar_constrained':(tuple(model['cvar']),'annual')}
    if name in fixed:
        return StaticPolicy(name,*fixed[name],interval=config['periodic_interval'],threshold=config['rebalance_threshold'])
    if name not in POLICY_NAMES:
        raise ValueError('Unknown policy: '+name)
    return GatedPolicy(name,model,params,config,adaptive=name in ('adaptive','adaptive_review'),
                       fixed=name=='balanced_gate',review=name=='adaptive_review')
