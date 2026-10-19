"""Float64 pathwise differentiation with empirical risk constraints and portable checkpoints."""
import time
import numpy as np
from .neural_core import NEURAL_NAMES,NEURAL_FEATURES,neural_data,normalizer,array_digest


def torch_runtime():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def tensor_data(data):
    torch=torch_runtime()
    return {key:torch.tensor(value,dtype=torch.float64) for key,value in data.items() if key!='scenario_id'}


def tail_mean(values,alpha=.9):
    torch=torch_runtime()
    descending=torch.sort(values,descending=True).values
    mass=len(values)*(1-alpha);whole=int(np.floor(mass));fraction=mass-whole
    return (descending[:whole].sum()+fraction*descending[whole])/mass


def network(name,config,seed):
    torch=torch_runtime();torch.manual_seed(seed)
    inputs=len(NEURAL_FEATURES) if name=='neural_portfolio' else 13
    if name=='neural_linear':net=torch.nn.Sequential(torch.nn.Linear(inputs,3))
    else:
        hidden=config['neural_hidden']
        net=torch.nn.Sequential(torch.nn.Linear(inputs,hidden),torch.nn.Tanh(),
            torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,3))
    net=net.double()
    with torch.no_grad():
        net[-1].weight.zero_();net[-1].bias.copy_(torch.tensor([1.,0.,-1.],dtype=torch.float64))
    return net


def serialize(net):
    torch=torch_runtime()
    return dict(layers=[dict(weight=layer.weight.detach().numpy().tolist(),bias=layer.bias.detach().numpy().tolist())
        for layer in net if isinstance(layer,torch.nn.Linear)])


def restore(member):
    torch=torch_runtime();layers=[]
    for index,item in enumerate(member['layers']):
        weight=torch.tensor(item['weight'],dtype=torch.float64);bias=torch.tensor(item['bias'],dtype=torch.float64)
        layer=torch.nn.Linear(weight.shape[1],weight.shape[0]).double()
        with torch.no_grad():layer.weight.copy_(weight);layer.bias.copy_(bias)
        layers.append(layer)
        if index+1<len(member['layers']):layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def rollout(nets,name,data,mean,scale,mu,cov,config,horizon,reference=False):
    torch=torch_runtime();n=data['nominal'].shape[0]
    costs=torch.tensor([config['cost_rate'],2*config['cost_rate']],dtype=torch.float64).repeat_interleave(n)
    factors=((1+data['nominal'])/(1+data['inflation'][:,:,None])).repeat((2,1,1))
    market=data['history'].repeat((2,1,1))
    anchor=torch.tensor([.5,.35,.15],dtype=torch.float64)
    targets=torch.tensor(config['neural_targets'],dtype=torch.float64)
    h=torch.zeros((2*n,3),dtype=torch.float64);h[:,2]=config['initial_wealth']
    peak=torch.full((2*n,),config['initial_wealth'],dtype=torch.float64)
    mdd=torch.zeros_like(peak);pending=torch.zeros(2*n,dtype=torch.bool)
    total_cost=torch.zeros_like(peak);trades=torch.zeros_like(peak)
    features=list(NEURAL_FEATURES) if name=='neural_portfolio' else list(range(13))
    for t in range(horizon):
        value=h.sum(dim=1);p=h/value[:,None];drawdown=1-value/peak
        if t==0:
            target=anchor.expand_as(h);trade=torch.ones(2*n,dtype=torch.bool)
        elif reference:
            target=anchor.expand_as(h);trade=torch.ones(2*n,dtype=torch.bool)
        else:
            x=torch.cat((p[:,:2],drawdown[:,None],mdd[:,None],market[:,t,:],costs[:,None],
                pending.to(torch.float64)[:,None],torch.full((2*n,1),(horizon-t)/30,dtype=torch.float64)),dim=1)
            z=torch.clamp((x-mean)/scale,-5,5)[:,features]
            target=sum(torch.softmax(net(z),dim=1)@targets for net in nets)/len(nets)
            target_u=1+target@mu-.5*config['neural_gate_gamma']*torch.einsum('bi,ij,bj->b',target,cov,target)
            current_u=1+p@mu-.5*config['neural_gate_gamma']*torch.einsum('bi,ij,bj->b',p,cov,p)
            benefit=(target_u-current_u)/target_u
            fraction=costs*.5*torch.abs(target-p).sum(dim=1)
            invalid=(~torch.isfinite(benefit))|(target_u<=0)
            pending=pending|(drawdown>=config['review_drawdown'])|invalid
            trade=(~pending)&(benefit>fraction+config['neural_buffer'])
        moved=.5*torch.abs(value[:,None]*target-h).sum(dim=1)
        fee=costs*moved
        fee=torch.where(trade,fee,torch.zeros_like(fee))
        h=torch.where(trade[:,None],(value-fee)[:,None]*target,h)
        total_cost=total_cost+fee
        if t:trades=trades+(trade&(moved>1e-10*config['initial_wealth'])).to(torch.float64)
        post=h.sum(dim=1);peak=torch.maximum(peak,post);mdd=torch.maximum(mdd,1-post/peak)
        h=h*factors[:,t,:]
        wealth=h.sum(dim=1);peak=torch.maximum(peak,wealth);mdd=torch.maximum(mdd,1-wealth/peak)
    wealth=h.sum(dim=1)
    log_growth=torch.log(wealth/config['initial_wealth'])/horizon
    return dict(wealth=wealth,log_growth=log_growth,mdd=mdd,total_real_cost=total_cost,
                trade_rate=trades/max(1,horizon-1),review=pending)


def summarized(values,n):
    torch=torch_runtime();rows=[]
    for slot in range(2):
        g=values['log_growth'][slot*n:(slot+1)*n];d=values['mdd'][slot*n:(slot+1)*n]
        rows.append(dict(mean_log_growth=float(g.mean().detach()),weak_log_growth=float((-tail_mean(-g)).detach()),
                         cvar_drawdown=float(tail_mean(d).detach())))
    return rows


def ensemble_constraints(nets,name,data,mean,scale,mu,cov,reference,config):
    torch=torch_runtime();rows=[];n=data['nominal'].shape[0]
    with torch.no_grad():
        for horizon in config['horizons']:
            metrics=summarized(rollout(nets,name,data,mean,scale,mu,cov,config,horizon),n)
            for cost_slot,cost in enumerate((config['cost_rate'],2*config['cost_rate'])):
                ref=reference[str(horizon)][cost_slot];row=metrics[cost_slot]
                risk_bound=ref['cvar_drawdown']-config['neural_risk_improvement']
                growth_bound=ref['weak_log_growth']-config['neural_weak_growth_margin']
                feasible=row['cvar_drawdown']<=risk_bound+1e-10 and row['weak_log_growth']>=growth_bound-1e-10
                rows.append(dict(horizon=horizon,cost_rate=cost,**row,risk_bound=risk_bound,weak_growth_bound=growth_bound,feasible=bool(feasible)))
    return rows


def fit_neural_models(scenarios,base_model,config):
    torch=torch_runtime();started=time.perf_counter()
    raw=neural_data(scenarios,config);data=tensor_data(raw);center,scales=normalizer(raw,config)
    mean=torch.tensor(center,dtype=torch.float64);scale=torch.tensor(scales,dtype=torch.float64)
    mu=torch.tensor(base_model['mu'],dtype=torch.float64);cov=torch.tensor(base_model['cov'],dtype=torch.float64)
    n=len(scenarios);reference={}
    with torch.no_grad():
        for horizon in config['horizons']:
            reference[str(horizon)]=summarized(rollout([],NEURAL_NAMES[0],data,mean,scale,mu,cov,config,horizon,reference=True),n)
    fitted={}
    for name in NEURAL_NAMES:
        checkpoints={str(epoch):dict(members=[]) for epoch in config['neural_checkpoints']};trace=[]
        for seed in config['neural_seeds']:
            net=network(name,config,seed)
            optimizer=torch.optim.Adam(net.parameters(),lr=config['neural_learning_rate'],weight_decay=config['neural_weight_decay'])
            dual=torch.full((2,len(config['horizons']),2),.1,dtype=torch.float64)
            for epoch in range(1,config['neural_epochs']+1):
                slot=(epoch-1)%len(config['horizons']);horizon=config['horizons'][slot]
                values=rollout([net],name,data,mean,scale,mu,cov,config,horizon)
                violations=[];raw_stats=[]
                for cost_slot in range(2):
                    g=values['log_growth'][cost_slot*n:(cost_slot+1)*n]
                    risk=tail_mean(values['mdd'][cost_slot*n:(cost_slot+1)*n]);weak=-tail_mean(-g)
                    ref=reference[str(horizon)][cost_slot]
                    violation=torch.stack(((risk-(ref['cvar_drawdown']-config['neural_risk_improvement']))/.01,
                        ((ref['weak_log_growth']-config['neural_weak_growth_margin'])-weak)/.001))
                    violations.append(violation);raw_stats.append((float(risk.detach()),float(weak.detach())))
                violation=torch.stack(violations)
                loss=-100*values['log_growth'].mean()+(dual[:,slot,:]*violation).sum()/2
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite neural objective')
                optimizer.zero_grad();loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),config['neural_gradient_clip'])
                if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite neural gradient')
                optimizer.step()
                dual[:,slot,:]=torch.clamp(dual[:,slot,:]+config['neural_dual_rate']*violation.detach(),0,config['neural_dual_cap'])
                if epoch==1 or epoch%5==0 or str(epoch) in checkpoints:
                    for cost_slot in range(2):
                        trace.append(dict(seed=seed,epoch=epoch,horizon=horizon,cost_rate=config['cost_rate']*(cost_slot+1),
                            loss=float(loss.detach()),gradient_norm=float(gradient),cvar_drawdown=raw_stats[cost_slot][0],
                            weak_log_growth=raw_stats[cost_slot][1],risk_violation=float(violation[cost_slot,0].detach()),
                            growth_violation=float(violation[cost_slot,1].detach()),risk_dual=float(dual[cost_slot,slot,0]),
                            growth_dual=float(dual[cost_slot,slot,1])))
                if str(epoch) in checkpoints:checkpoints[str(epoch)]['members'].append(serialize(net))
        for checkpoint in checkpoints.values():
            nets=[restore(member) for member in checkpoint['members']]
            checkpoint['training_constraints']=ensemble_constraints(nets,name,data,mean,scale,mu,cov,reference,config)
            checkpoint['training_feasible']=all(r['feasible'] for r in checkpoint['training_constraints'])
        fitted[name]=dict(train_ids=sorted(raw['scenario_id'].tolist()),dataset_sha256=array_digest(raw),
            mean=center.tolist(),scale=scales.tolist(),features=list(NEURAL_FEATURES) if name=='neural_portfolio' else list(range(13)),
            targets=config['neural_targets'],seeds=config['neural_seeds'],checkpoints=checkpoints,reference=reference,trace=trace,
            parameter_count=sum(p.numel() for p in net.parameters()),framework_version=torch.__version__)
    for fit in fitted.values():fit['joint_fit_seconds']=time.perf_counter()-started
    return fitted
