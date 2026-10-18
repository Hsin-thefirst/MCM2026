"""Fixed A-continuation and conditional bootstrap-value research extensions."""
from dataclasses import replace
import numpy as np
from safeinvest.contracts import Action, Decision
from safebench.plan_c import batch_a
from efficient_fusion import EfficientFusion
from fusion_policy import tail


def path_score(terminal, losses, maximum, horizon, settings):
    risk = tail(losses)
    p90 = np.quantile(losses,.9,axis=1)
    q = np.log(terminal).mean(axis=1)/horizon-settings['drawdown_penalty']*(risk-maximum)
    q -= settings['risk_barrier']*np.maximum(0.,p90-p90[1]-settings['risk_slack'])
    return q,risk


def hold_paths(targets,current,drawdown,maximum,future,cost):
    fees = cost*.5*np.abs(targets-current).sum(axis=1)
    gross = np.cumprod(1+future,axis=1)
    wealth = np.einsum('ak,stk->ast',targets,gross)*(1-fees[:,None,None])
    marks = np.concatenate((np.broadcast_to(1-fees[:,None,None],(len(targets),len(future),1)),wealth),axis=2)
    peaks = np.maximum(np.maximum.accumulate(marks,axis=2),1/(1-drawdown))
    losses = np.maximum((1-marks/peaks).max(axis=2),maximum)
    return wealth[:,:,-1],losses,fees


def weighted_bootstrap_risk(losses,counts):
    """Exact empirical ES and linear quantiles without repeated sample sorting."""
    order = np.argsort(losses,axis=1)[:,::-1]
    descending = np.take_along_axis(losses,order,axis=1)
    weights = counts[:,order]
    cumulative = weights.cumsum(axis=2)
    mass = losses.shape[1]*(1-.9)
    selected = np.minimum(weights,np.maximum(0.,mass-(cumulative-weights)))
    risk = (selected*descending[None,:,:]).sum(axis=2)/mass
    # A descending .9-quantile uses rank .1*(n-1).
    position = (1-.9)*(losses.shape[1]-1)
    lower = int(np.floor(position)); upper = int(np.ceil(position))
    lo = np.argmax(cumulative>lower,axis=2)
    hi = np.argmax(cumulative>upper,axis=2)
    values = np.broadcast_to(descending,weights.shape)
    a = np.take_along_axis(values,lo[:,:,None],axis=2)[:,:,0]
    b = np.take_along_axis(values,hi[:,:,None],axis=2)[:,:,0]
    return risk,a+(position-lower)*(b-a)


def uncertainty_values(targets,current,drawdown,maximum,future,horizon,cost,settings,counts):
    terminal,losses,fees = hold_paths(targets,current,drawdown,maximum,future,cost)
    q,risk = path_score(terminal,losses,maximum,horizon,settings)
    es,p90 = weighted_bootstrap_risk(losses,counts)
    values = counts@np.log(terminal).T/len(future)/horizon-settings['drawdown_penalty']*(es-maximum)
    values -= settings['risk_barrier']*np.maximum(0.,p90-p90[:,1,None]-settings['risk_slack'])
    sd = (values-values[:,0,None]).std(axis=0,ddof=1)
    return q-sd,fees,risk,dict(raw_values=q.tolist(),gain_sd=sd.tolist())


def continuation_values(policy,c,targets):
    future = policy.future[:,c.period-1:policy.horizon]
    current = np.asarray(c.holdings)/sum(c.holdings)
    dd = max(0.,1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
    holdings = np.broadcast_to(current,(len(targets),len(future),3)).copy()
    peak = np.full(holdings.shape[:2],1/(1-dd))
    losses = np.full(peak.shape,c.historical_max_drawdown)
    pending = np.zeros(peak.shape,dtype=bool)
    cumulative_fees = np.zeros(peak.shape)
    first_fees = c.transaction_cost_rate*.5*np.abs(targets-current).sum(axis=1)
    for step in range(future.shape[1]):
        wealth = holdings.sum(axis=2)
        weights = holdings/wealth[:,:,None]
        if step == 0:
            target = np.broadcast_to(targets[:,None,:],holdings.shape)
        else:
            pending |= 1-wealth/peak >= policy.adviser.review_threshold
            advice,invalid = batch_a(weights,policy.adviser,c.transaction_cost_rate)
            pending |= invalid
            target = np.where(pending[:,:,None],weights,advice)
        fee = c.transaction_cost_rate*.5*np.abs(target-weights).sum(axis=2)*wealth
        cumulative_fees += fee
        post = wealth-fee
        losses = np.maximum(losses,1-post/peak)
        peak = np.maximum(peak,post)
        holdings = post[:,:,None]*target*(1+future[None,:,step,:])
        wealth = holdings.sum(axis=2)
        losses = np.maximum(losses,1-wealth/peak)
        peak = np.maximum(peak,wealth)
    q,risk = path_score(wealth,losses,c.historical_max_drawdown,policy.horizon,policy.fusion)
    return q,first_fees,risk,dict(mean_continuation_fees=cumulative_fees.mean(axis=1).tolist())


class ValueFusion(EfficientFusion):
    def __init__(self,name,model,params,config,horizon,settings,rule,variant):
        super().__init__(name,model,params,config,horizon,settings,rule)
        if variant not in ['a_continuation','uncertainty']: raise ValueError('Unknown value extension')
        self.variant = variant
        if variant == 'uncertainty':
            n = len(self.future)
            indices = np.random.default_rng(20261021).integers(0,n,(200,n))
            self.bootstrap_counts = np.stack([np.bincount(row,minlength=n) for row in indices])

    def _decide(self,c):
        if self.use_a: return super()._decide(c)
        self.last_diagnostics = None
        advice = self.adviser._decide(c)
        if advice.action == Action.HUMAN_REVIEW: return advice
        current = np.asarray(c.holdings)/sum(c.holdings)
        targets = self.candidates(c,advice)
        dd = max(0.,1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
        if self.variant == 'a_continuation':
            q,fees,risk,extra = continuation_values(self,c,targets)
        else:
            q,fees,risk,extra = uncertainty_values(targets,current,dd,c.historical_max_drawdown,
                self.future[:,c.period-1:self.horizon],self.horizon,c.transaction_cost_rate,self.fusion,self.bootstrap_counts)
        if not np.isfinite(q).all():
            return Decision(Action.HUMAN_REVIEW,('VALUE_INVALID',),
                'The estimate is unreliable. Existing investments remain in place while a human adviser reviews your plan.')
        benefit = fees+(1-fees)*(-np.expm1(np.clip(-self.horizon*(q-q[0]),-100,100)))
        allowed = benefit>fees+self.buffer
        allowed[0] = True
        chosen = int(np.argmax(np.where(allowed,q,-np.inf)))
        self.last_diagnostics = dict(values=q.tolist(),costs=fees.tolist(),benefits=benefit.tolist(),
            targets=targets.tolist(),chosen=chosen,tail_drawdowns=risk.tolist(),required_buffer=self.buffer,
            historical_max_drawdown=c.historical_max_drawdown,move_cap=self.fusion['move_cap'],variant=self.variant,**extra)
        if chosen == 0:
            return Decision(Action.KEEP,('VALUE_HOLD',),
                'Keep existing investments: projected improvement does not cover trading costs and the safety margin.',
                estimated_benefit=0.,estimated_cost=0.)
        decision = self._trade(c,targets[chosen],float(benefit[chosen]),float(fees[chosen]),'VALUE_NET_IMPROVEMENT')
        return replace(decision,explanation='This adjustment improves the scenario-based estimate after costs and a safety margin. Projections are uncertain; investment losses remain possible.')
