"""Fixed validation-budget penalties for the last declared research round."""
from dataclasses import replace
import numpy as np
from safeinvest.contracts import Action,Decision
from fusion_value_extensions import ValueFusion,hold_paths,path_score,weighted_bootstrap_risk
from fusion_policy import tail


def bootstrap_log_risk(losses,counts):
    order=np.argsort(losses,axis=1)[:,::-1]
    ordered=np.take_along_axis(-np.log1p(-losses),order,axis=1)
    weights=counts[:,order]
    cumulative=weights.cumsum(axis=2)
    mass=losses.shape[1]*(1-.9)
    selected=np.minimum(weights,np.maximum(0.,mass-cumulative+weights))
    return (selected*ordered[None,:,:]).sum(axis=2)/mass


def budget_values(targets,current,drawdown,maximum,future,horizon,cost,settings,counts,budget,uncertainty,coefficient=1.):
    terminal,losses,fees=hold_paths(targets,current,drawdown,maximum,future,cost)
    raw,risk=path_score(terminal,losses,maximum,horizon,settings)
    log_risk=tail(-np.log1p(-losses))
    excess=np.maximum(0.,log_risk-budget)
    q=raw-coefficient*excess/horizon
    sd=np.zeros(len(targets))
    if uncertainty:
        es,p90=weighted_bootstrap_risk(losses,counts)
        values=counts@np.log(terminal).T/len(future)/horizon-settings['drawdown_penalty']*(es-maximum)
        values-=settings['risk_barrier']*np.maximum(0.,p90-p90[:,1,None]-settings['risk_slack'])
        values-=coefficient*np.maximum(0.,bootstrap_log_risk(losses,counts)-budget)/horizon
        sd=(values-values[:,0,None]).std(axis=0,ddof=1)
    return q-sd,fees,risk,dict(raw_values=raw.tolist(),budget_values=q.tolist(),gain_sd=sd.tolist(),
        log_drawdown_es=log_risk.tolist(),validation_budget=budget,budget_excess=excess.tolist(),coefficient=coefficient)


class BudgetFusion(ValueFusion):
    def __init__(self,name,model,params,config,horizon,settings,rule,variant):
        super().__init__(name,model,params,config,horizon,settings,rule,'uncertainty')
        if variant not in ['budget_uncertainty','budget_mean']: raise ValueError('Unknown budget rule')
        self.variant=variant
        entry=model['plan_a']['entries'][f"balanced:{horizon}:{params['risk_ratio']:g}"]
        self.validation_budget=entry.get('validation_budget',entry['risk_budget'])

    def _decide(self,c):
        if self.use_a: return super()._decide(c)
        self.last_diagnostics=None
        advice=self.adviser._decide(c)
        if advice.action==Action.HUMAN_REVIEW: return advice
        current=np.asarray(c.holdings)/sum(c.holdings)
        targets=self.candidates(c,advice)
        dd=max(0.,1-sum(c.holdings)/c.cumulative_inflation/c.real_peak)
        q,fees,risk,extra=budget_values(targets,current,dd,c.historical_max_drawdown,
            self.future[:,c.period-1:self.horizon],self.horizon,c.transaction_cost_rate,self.fusion,
            self.bootstrap_counts,self.validation_budget,self.variant=='budget_uncertainty')
        if not np.isfinite(q).all():
            return Decision(Action.HUMAN_REVIEW,('BUDGET_INVALID',),
                'The estimate is unreliable. Existing investments remain in place while a human adviser reviews your plan.')
        benefit=fees+(1-fees)*(-np.expm1(np.clip(-self.horizon*(q-q[0]),-100,100)))
        allowed=benefit>fees+self.buffer;allowed[0]=True
        chosen=int(np.argmax(np.where(allowed,q,-np.inf)))
        self.last_diagnostics=dict(values=q.tolist(),costs=fees.tolist(),benefits=benefit.tolist(),targets=targets.tolist(),
            chosen=chosen,tail_drawdowns=risk.tolist(),required_buffer=self.buffer,
            historical_max_drawdown=c.historical_max_drawdown,move_cap=self.fusion['move_cap'],variant=self.variant,**extra)
        if chosen==0:
            return Decision(Action.KEEP,('BUDGET_HOLD',),
                'Keep existing investments: the projected improvement does not cover costs and the safety margin.',
                estimated_benefit=0.,estimated_cost=0.)
        decision=self._trade(c,targets[chosen],float(benefit[chosen]),float(fees[chosen]),'BUDGET_NET_VALUE')
        return replace(decision,explanation='This adjustment accounts for the projected risk budget and trading costs. Estimates are uncertain and do not guarantee a limit on investment losses.')
