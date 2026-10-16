"""Smooth drawdown-dependent risk preferences on a fine long-only grid."""
import time
from functools import lru_cache
import numpy as np
from safeinvest.contracts import Action, Decision
from .policies import PORTFOLIOS, grid

CUSHION_NAMES=('cushion_static','cushion_gate')
CALIBRATED_NAMES=('calibrated_fixed','calibrated_gate')


def enabled(config):
    return config.get('cushion_research',{}).get('enabled',False)


@lru_cache(maxsize=8)
def candidate_grid(step):
    return grid(bounds=((0.,.65),(0.,.7),(.1,1.)),step=step)


class CushionPolicy:
    requires_benefit_gate=True

    def __init__(self,name,model,params,config,horizon):
        self.name=name; self.horizon=horizon
        self.settings=config['cushion_research']
        self.gamma_at_peak=params.get('gamma',self.settings['gamma_at_peak'])
        self.mu=np.array(model['mu']); self.cov=np.array(model['cov'])
        prior_weight=0. if name in CALIBRATED_NAMES else self.settings.get('strategic_prior_weight',0.)
        if prior_weight and np.isfinite(self.mu).all() and np.isfinite(self.cov).all():
            balanced=np.array(PORTFOLIOS['balanced'])
            implied=self.settings['gamma_at_peak']*self.cov@balanced
            implied+=self.mu@balanced-implied@balanced
            self.mu=(1-prior_weight)*self.mu+prior_weight*implied
        self.loss_budget=params.get('loss_budget',self.settings['loss_budgets'][0])
        self.initial_target=PORTFOLIOS['balanced']; self.buffer=self.settings['buffer']
        # A ceiling on risky exposures allows a continuous path to bills.
        self.candidates=candidate_grid(self.settings['grid_step'])
        self.candidate_mean=self.candidates@self.mu
        self.candidate_variance=np.einsum('ni,ij,nj->n',self.candidates,self.cov,self.candidates)
        self.inference_seconds=0.; self.decision_calls=0

    def decide(self,c):
        began=time.perf_counter()
        try:
            return self._decide(c)
        finally:
            self.inference_seconds+=time.perf_counter()-began; self.decision_calls+=1

    def _decide(self,c):
        wealth=sum(c.holdings)/c.cumulative_inflation
        drawdown=max(0.,1-wealth/c.real_peak)
        if drawdown>=self.settings['suspension_drawdown'] or not np.isfinite(self.mu).all() or not np.isfinite(self.cov).all():
            code='CUSHION_SUSPENSION' if drawdown>=self.settings['suspension_drawdown'] else 'CUSHION_INVALID_ESTIMATE'
            return Decision(Action.HUMAN_REVIEW,(code,),
                'The automated risk assessment requires human review. We will hold existing investments while your plan is reassessed. This suspension does not guarantee a limit on losses.')
        gamma=self.gamma_at_peak
        if self.name=='cushion_gate':
            effective=min(drawdown,self.loss_budget-.02)
            ratio=self.loss_budget*(1-effective)/max(self.loss_budget-effective,.02)
            gamma*=1+self.settings.get('risk_response_strength',1.)*(ratio-1)
        weights=np.array(c.holdings)/sum(c.holdings)
        targets=np.vstack((weights,c.target_weights,self.candidates))
        small=targets[:2]
        utility=1+np.r_[small@self.mu,self.candidate_mean]-.5*gamma*np.r_[
            np.einsum('ni,ij,nj->n',small,self.cov,small),self.candidate_variance]
        fee=c.transaction_cost_rate*.5*np.abs(targets-weights).sum(axis=1)
        if not np.isfinite(utility).all() or np.any(utility<=0):
            return Decision(Action.HUMAN_REVIEW,('CUSHION_INVALID_VALUE',),
                'Our risk-adjusted allocation estimates are unreliable. We will hold existing investments while a human adviser reviews your plan.')
        benefit=(utility-utility[0])/utility
        allowed=benefit>fee+self.buffer; allowed[0]=True
        selected=int(np.argmax(np.where(allowed,(1-fee)*utility,-np.inf)))
        codes=('CUSHION_DYNAMIC' if self.name=='cushion_gate' else 'CUSHION_STATIC',
               'LEGAL_ACTIONS_'+str(int(allowed.sum())),f'RISK_AVERSION_{gamma:.4f}')
        if selected==0:
            return Decision(Action.KEEP,codes,'Keep your holdings: no estimated risk-adjusted improvement exceeds transaction costs and our safety margin.',estimated_benefit=0.,estimated_cost=0.)
        target=tuple(targets[selected])
        action=Action.REBALANCE if np.allclose(target,c.target_weights,atol=1e-10,rtol=0) else Action.CHANGE_TARGET
        return Decision(action,codes+('BENEFIT_EXCEEDS_COST',),
            'Adjust the allocation because the estimated risk-adjusted benefit exceeds trading costs and our safety margin. Observed purchasing-power losses affect our risk assessment. This is not a loss guarantee.',
            target,float(benefit[selected]),float(fee[selected]))
