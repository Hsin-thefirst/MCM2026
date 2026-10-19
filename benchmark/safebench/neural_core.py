"""Portable neural proposals with the same explicit cost gate and absorbing review."""
import time
import numpy as np
from safeinvest.contracts import Action, Decision, validate_weights
from .learning_core import state_features, array_digest
from .scenario_utility import REVIEW_EXPLANATION

NEURAL_NAMES=('neural_linear','neural_mlp','neural_portfolio')
NEURAL_FEATURES=(0,1,2,3,10,11,12)


def neural_data(scenarios,config):
    from .learning_core import market_features
    ids=[s.scenario_id for s in scenarios]
    if not ids or len(ids)!=len(set(ids)) or any(not 1<=i<=120 for i in ids):
        raise PermissionError('Neural fitting requires unique development IDs')
    nominal=[];inflation=[];history=[]
    for scenario in scenarios:
        if len(scenario.observations)!=30:
            raise ValueError('Neural training requires complete 30-period development paths')
        rows=[];rates=[];features=[]
        for index,o in enumerate(scenario.observations):
            if o.scenario_id!=scenario.scenario_id or o.period!=index+1:
                raise ValueError('Invalid neural observation sequence')
            if not all(np.isfinite(v) and v>-1 for v in (*o.nominal_returns,o.inflation)):
                raise ValueError('Invalid neural training observation')
            rows.append(o.nominal_returns);rates.append(o.inflation)
            features.append(market_features(scenario.observations[:index]) if index else [0.]*6)
        nominal.append(rows);inflation.append(rates);history.append(features)
    return dict(scenario_id=np.array(ids,dtype=np.int64),nominal=np.array(nominal,dtype=np.float64),
                inflation=np.array(inflation,dtype=np.float64),history=np.array(history,dtype=np.float64))


def normalizer(data,config):
    mean=np.array([.4,.3,.15,.25,0,0,0,0,0,0,1.5*config['cost_rate'],0,.5],dtype=float)
    scale=np.array([.25,.2,.15,.25,1,1,1,1,1,1,config['cost_rate'],1,.5],dtype=float)
    history=data['history'][:,1:,:].reshape(-1,6)
    mean[4:10]=history.mean(axis=0)
    scale[4:10]=np.maximum(history.std(axis=0),.01)
    return mean,scale


def member_target(member,x,targets):
    value=np.atleast_2d(x)
    for index,layer in enumerate(member['layers']):
        value=value@np.array(layer['weight']).T+np.array(layer['bias'])
        if index+1<len(member['layers']):value=np.tanh(value)
    value=np.exp(value-value.max(axis=1,keepdims=True));value/=value.sum(axis=1,keepdims=True)
    return value@np.array(targets)


def neural_target(fitted,checkpoint,x):
    x=np.atleast_2d(x)
    standardized=np.clip((x-np.array(fitted['mean']))/np.array(fitted['scale']),-5,5)
    standardized=standardized[:,fitted['features']]
    members=fitted['checkpoints'][str(checkpoint)]['members']
    return sum(member_target(member,standardized,fitted['targets']) for member in members)/len(members)


def neural_gate(current,target,mu,cov,config,cost):
    gamma=config['neural_gate_gamma']
    target_u=1+np.array(target)@mu-.5*gamma*np.einsum('...i,ij,...j->...',target,cov,target)
    current_u=1+np.array(current)@mu-.5*gamma*np.einsum('...i,ij,...j->...',current,cov,current)
    benefit=(target_u-current_u)/target_u
    fee=cost*.5*np.abs(np.array(target)-current).sum(axis=-1)
    return benefit,fee,target_u


def save_neural_data(scenarios,config,model,path):
    data=neural_data(scenarios,config)
    if any(fit['dataset_sha256']!=array_digest(data) for fit in model['neural'].values()):
        raise ValueError('Neural dataset mismatch')
    with path.open('xb') as handle:np.savez_compressed(handle,**data)


class NeuralPolicy:
    requires_benefit_gate=True

    def __init__(self,name,model,params,config,horizon):
        if name not in NEURAL_NAMES or horizon is None or not 2<=horizon<=30:
            raise ValueError('Neural policy needs an explicit horizon')
        self.name=name;self.fitted=model['neural'][name];self.checkpoint=params['checkpoint']
        self.config=config;self.horizon=horizon;self.initial_target=(.5,.35,.15)
        self.mu=np.array(model['mu']);self.cov=np.array(model['cov'])
        self.buffer=config['neural_buffer'];self.maximum_drawdown=0.;self.last_period=1
        self.inference_seconds=0.;self.inference_calls=0

    def decide(self,context):
        started=time.perf_counter()
        try:return self._decide(context)
        finally:
            self.inference_seconds+=time.perf_counter()-started;self.inference_calls+=1

    def _decide(self,context):
        if context.period!=self.last_period+1 or len(context.history)!=context.period-1:
            raise ValueError('Neural policy requires sequential causal contexts')
        self.last_period=context.period
        value=sum(context.holdings)
        drawdown=1-value/context.cumulative_inflation/context.real_peak
        self.maximum_drawdown=max(self.maximum_drawdown,drawdown,.85*context.transaction_cost_rate)
        if drawdown>=self.config['review_drawdown']:
            return Decision(Action.HUMAN_REVIEW,('DRAWDOWN_ALERT',),REVIEW_EXPLANATION)
        current=np.array(context.holdings)/value
        x=np.r_[state_features(current,drawdown,self.maximum_drawdown,context.history,
                    context.transaction_cost_rate,context.review_pending),(self.horizon-context.period+1)/30]
        target=neural_target(self.fitted,self.checkpoint,x)[0]
        benefit,fee,utility=neural_gate(current,target,self.mu,self.cov,self.config,context.transaction_cost_rate)
        if not np.isfinite(target).all() or not np.isfinite(benefit) or utility<=0:
            return Decision(Action.HUMAN_REVIEW,('INVALID_ESTIMATE',),
                'The allocation estimate is unreliable. A human adviser should review your plan. We will hold your existing investments while review is pending.')
        validate_weights(tuple(target))
        if benefit>fee+self.buffer:
            self.maximum_drawdown=max(self.maximum_drawdown,1-(1-drawdown)*(1-fee))
            action=Action.REBALANCE if np.allclose(target,context.target_weights,atol=1e-10,rtol=0) else Action.CHANGE_TARGET
            return Decision(action,('NEURAL_BENEFIT_EXCEEDS_COST',),
                'This allocation reflects your remaining horizon, portfolio risk and past market information. Its estimated risk-adjusted benefit exceeds the trading cost and safety margin. Losses remain possible.',
                tuple(target),float(benefit),float(fee))
        return Decision(Action.KEEP,('NEURAL_BENEFIT_BELOW_COST',),
            'The estimated benefit does not justify the trading cost and safety margin.',
            estimated_benefit=float(benefit),estimated_cost=float(fee))
