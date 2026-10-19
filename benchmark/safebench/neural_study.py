"""Nested checkpoint selection and auditable reporting for constrained neural policies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from safeinvest.contracts import validate_weights
from .neural_core import NEURAL_NAMES,save_neural_data
from .learning_study import evaluate_learning,learning_intervals


def validate_neural_config(config):
    if config.get('study_kind')!='neural_path_control':raise ValueError('Wrong neural study kind')
    if not config['neural_seeds'] or len(set(config['neural_seeds']))!=len(config['neural_seeds']):
        raise ValueError('Invalid neural training seeds')
    checkpoints=config['neural_checkpoints']
    if not checkpoints or checkpoints!=sorted(set(checkpoints)) or checkpoints[-1]!=config['neural_epochs'] or checkpoints[0]<1:
        raise ValueError('Invalid neural checkpoints')
    if len(config['neural_targets'])!=3:raise ValueError('Exactly three neural proposal vertices required')
    for target in config['neural_targets']:
        validate_weights(target)
        if not .2-1e-12<=target[0]<=.5+1e-12:raise ValueError('Neural equity target outside declared range')
    for key in ('neural_hidden','neural_epochs'):
        if not isinstance(config[key],int) or config[key]<1:raise ValueError('Invalid neural architecture/budget')
    for key in ('neural_learning_rate','neural_dual_rate','neural_dual_cap','neural_gradient_clip',
                'neural_risk_improvement','neural_weak_growth_margin','neural_growth_margin','neural_p10_margin','neural_p90_improvement'):
        if not np.isfinite(config[key]) or config[key]<=0:raise ValueError('Invalid neural optimization value')
    if not {5,6,15,25,30}<=set(config['horizons']):raise ValueError('Customer horizons missing')
    if not any(np.isclose(c,2*config['cost_rate']) for c in config['sensitivity_costs']):raise ValueError('Doubled cost missing')


def select_neural_trials(scores,reference,config):
    ref=reference.set_index('scenario_id').sort_index();rows=[];params={}
    if ref.index.duplicated().any():raise ValueError('Unpaired neural reference')
    for name in NEURAL_NAMES:
        options=[]
        for trial,checkpoint in enumerate(config['neural_checkpoints']):
            group=scores[(scores.policy==name)&(scores.trial==trial)].set_index('scenario_id').sort_index()
            if not group.index.equals(ref.index) or group.index.duplicated().any():raise ValueError('Unpaired neural candidate')
            growth=float(group.real_cagr.median()-ref.real_cagr.median())
            p10=float(group.real_cagr.quantile(.1)-ref.real_cagr.quantile(.1))
            p90=float(group.real_max_drawdown.quantile(.9));ref_p90=float(ref.real_max_drawdown.quantile(.9))
            violation=max(0.,(-growth-config['neural_growth_margin'])/config['neural_growth_margin'],
                (-p10-config['neural_p10_margin'])/config['neural_p10_margin'],
                (p90-ref_p90+config['neural_p90_improvement'])/config['neural_p90_improvement'])
            row=dict(policy=name,trial=trial,checkpoint=checkpoint,growth_difference=growth,p10_difference=p10,
                p90_loss=p90,reference_p90_loss=ref_p90,p90_difference=p90-ref_p90,
                median_growth=float(group.real_cagr.median()),normalized_violation=violation,
                risk_feasible=violation<=1e-10,selected=False,selection_status='not_selected')
            rows.append(row);options.append(row)
        feasible=[r for r in options if r['risk_feasible']]
        chosen=min(feasible or options,key=lambda r:(0 if feasible else r['normalized_violation'],
                                                      -r['median_growth'],r['p90_loss'],r['checkpoint']))
        chosen['selected']=True;chosen['selection_status']='feasible' if chosen['risk_feasible'] else 'infeasible_diagnostic'
        params[name]=dict(checkpoint=chosen['checkpoint'],selection_trial=chosen['trial'],selection_status=chosen['selection_status'])
    return params,rows


def tune_neural(scenarios,config,seed,trace_path=None):
    from .data import folds
    from .policies import fit_model
    from .experiment import dump
    validate_neural_config(config)
    path=Path(trace_path) if trace_path is not None else None
    if path is not None and path.exists():raise FileExistsError(path)
    by_id={s.scenario_id:s for s in scenarios};collected=[];refs=[];audit=[]
    for inner,valid_ids in enumerate(folds(by_id,config['inner_folds'],seed)):
        train=[s for sid,s in by_id.items() if sid not in valid_ids];valid=[by_id[sid] for sid in valid_ids]
        model=fit_model(train,config)
        audit.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=valid_ids))
        if path is not None:
            model_dir=path.parent.parent/'tuning_models';model_dir.mkdir(parents=True,exist_ok=True)
            data_dir=path.parent.parent/'training_data';data_dir.mkdir(exist_ok=True)
            stem=path.name.split('.')[0]+f'_inner_{inner}'
            dump(model_dir/(stem+'.json'),model);save_neural_data(train,config,model,data_dir/(stem+'.npz'))
        reference,_=evaluate_learning(valid,model,{},config,names=[config['neural_reference']],horizons=[30],fold=inner,keep_records=False)
        reference['trial']=-1;refs.append(reference)
        for trial,checkpoint in enumerate(config['neural_checkpoints']):
            metrics,_=evaluate_learning(valid,model,{name:dict(checkpoint=checkpoint) for name in NEURAL_NAMES},
                config,names=NEURAL_NAMES,horizons=[30],fold=inner,keep_records=False)
            metrics['trial']=trial;collected.append(metrics)
    scores=pd.concat(collected,ignore_index=True);reference=pd.concat(refs,ignore_index=True)
    chosen,rows=select_neural_trials(scores,reference,config)
    if path is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        pd.concat([scores,reference],ignore_index=True).to_csv(path,index=False,compression={'method':'gzip','mtime':0})
    return chosen,rows,audit


def write_neural_analysis(out,config):
    from . import ROOT
    from .metrics import binomial_upper,paired_comparisons
    out=Path(out);metrics=pd.read_csv(out/'scenario_metrics.csv');trials=pd.read_csv(out/'tuning_trials.csv')
    intervals=learning_intervals(metrics,config,policy_names=NEURAL_NAMES,prefix='neural')
    intervals.to_csv(out/'neural_intervals.csv',index=False)
    fitted=json.loads((out/'fitted_models.json').read_text());constraints=[];training=[]
    for fold in fitted:
        for name,fit in fold['model']['neural'].items():
            for row in fit['trace']:training.append(dict(fold=fold['fold'],policy=name,**row))
            for checkpoint,model in fit['checkpoints'].items():
                for row in model['training_constraints']:
                    constraints.append(dict(fold=fold['fold'],policy=name,checkpoint=int(checkpoint),
                        selected=int(checkpoint)==fold['params'][name]['checkpoint'],**row))
    constraints=pd.DataFrame(constraints);constraints.to_csv(out/'neural_constraints.csv',index=False)
    pd.DataFrame(training).to_csv(out/'neural_training.csv',index=False)
    screens=[]
    for name in NEURAL_NAMES:
        chosen=trials[(trials.policy==name)&trials.selected]
        train=constraints[(constraints.policy==name)&constraints.selected]
        inner=bool(len(chosen)==config['outer_folds'] and chosen.risk_feasible.all())
        training_ok=bool(len(train)==config['outer_folds']*len(config['horizons'])*2 and train.feasible.all())
        bounds=bool(intervals[intervals.policy==name].passed.all())
        passed=inner and training_ok and bounds
        screens.append(dict(policy=name,feasible_folds=int(chosen.risk_feasible.sum()),inner_selection_feasible=inner,
            selected_training_feasible=training_ok,all_interval_conditions=bounds,all_checks_passed=passed,
            status='development_risk_improvement' if passed else 'not_established'))
    pd.DataFrame(screens).to_csv(out/'neural_screen.csv',index=False)
    profiles=json.loads((ROOT/'configs/profiles.json').read_text());risks=[]
    confidence=1-(1-config['suitability_confidence'])/(3+len(NEURAL_NAMES))
    for p in profiles:
        for name in NEURAL_NAMES:
            g=metrics[(metrics.policy==name)&(metrics.horizon==p['horizon'])&np.isclose(metrics.cost_rate,config['cost_rate'])]
            if not len(g) or g.scenario_id.duplicated().any():raise ValueError('Missing neural customer horizon')
            k=int((g.real_max_drawdown>p['loss_tolerance']).sum());upper=binomial_upper(k,len(g),confidence)
            liquidity_ok=p['liquidity']!='high';risk_ok=upper<=config['suitability_max_breach_probability']
            risks.append(dict(profile=p['profile_id'],policy=name,horizon=p['horizon'],tolerance=p['loss_tolerance'],
                n=len(g),breaches=k,breach_probability=k/len(g),confidence=confidence,upper_bound=upper,
                liquidity_ok=liquidity_ok,risk_pass=risk_ok,eligible=bool(liquidity_ok and risk_ok),
                median_growth=float(g.real_cagr.median()),diagnostic_only=True))
    pd.DataFrame(risks).to_csv(out/'neural_suitability.csv',index=False)
    subset=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])&
                   metrics.policy.isin(['balanced_annual_review',*NEURAL_NAMES])]
    paired_comparisons(subset,config['seed'],config['bootstrap'],baseline='balanced_annual_review').to_csv(
        out/'neural_review_comparisons.csv',index=False)


def generate_neural_report(out,config):
    from .report import plt,LABELS
    out=Path(out);summary=pd.read_csv(out/'summary.csv');intervals=pd.read_csv(out/'neural_intervals.csv')
    constraints=pd.read_csv(out/'neural_constraints.csv');training=pd.read_csv(out/'neural_training.csv')
    screen=pd.read_csv(out/'neural_screen.csv');trials=pd.read_csv(out/'tuning_trials.csv')
    risks=pd.read_csv(out/'neural_suitability.csv')
    zh={'neural_linear':'同约束线性策略','neural_mlp':'完整特征MLP','neural_portfolio':'仅持仓回撤MLP',
        'equal_annual':'官方等权','balanced_annual':'年度Balanced','balanced_annual_review':'年度Balanced加相同复核'}
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg','pdf'):fig.savefig(out/'figures'/(name+'.'+extension),dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for axis,endpoint,label in zip(axes,('median_growth','p10_growth','p90_loss'),
        ('Median real CAGR difference\n(pp / year)','P10 real CAGR difference\n(pp / year)','P90 drawdown difference\n(pp)')):
        group=intervals[(intervals.metric==endpoint)&np.isclose(intervals.cost_rate,config['cost_rate'])].set_index('policy')
        for y,name in enumerate(NEURAL_NAMES):
            row=group.loc[name];axis.plot(100*np.array([row.family_lower,row.family_upper]),[y,y],color='#287c8e',lw=2)
            axis.scatter(100*row.difference,y,color='#173e5b')
        axis.axvline(100*group.iloc[0].threshold,color='#ae5b32',ls='--');axis.axvline(0,color='gray',ls=':',lw=.8)
        axis.set_xlabel(label);axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(3),[LABELS[n] for n in NEURAL_NAMES]);save(fig,'neural_intervals')
    selected=constraints[constraints.selected].copy()
    selected['risk_violation']=(selected.cvar_drawdown-selected.risk_bound)/.01
    selected['growth_violation']=(selected.weak_growth_bound-selected.weak_log_growth)/.001
    matrix=np.array([[max(selected[(selected.policy==name)&(selected.fold==fold)][['risk_violation','growth_violation']].max())
                      for fold in range(config['outer_folds'])] for name in NEURAL_NAMES])
    fig,axis=plt.subplots(figsize=(9,3.5));axis.imshow(matrix,cmap='RdYlGn_r',vmin=-1,vmax=max(1,min(float(matrix.max()),10)),aspect='auto')
    for y in range(3):
        for fold in range(config['outer_folds']):
            axis.text(fold,y,f'{matrix[y,fold]:.2f}\n'+('FEASIBLE' if matrix[y,fold]<=1e-8 else 'INFEASIBLE'),ha='center',va='center',fontsize=8)
    axis.set_yticks(range(3),[LABELS[n] for n in NEURAL_NAMES]);axis.set_xticks(range(config['outer_folds']),[str(i+1) for i in range(config['outer_folds'])])
    axis.set_xlabel('Outer fold; maximum normalized training constraint violation');save(fig,'neural_constraints')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for name,color in zip(NEURAL_NAMES,('#287c8e','#ae5b32','#79549c')):
        block=training[(training.horizon==30)&np.isclose(training.cost_rate,config['cost_rate'])]
        group=block[block.policy==name].groupby('epoch')
        for axis,column,label in zip(axes,('risk_violation','growth_violation','gradient_norm'),
            ('Normalized MDD-CVaR violation','Normalized weak-growth violation','Gradient norm before clipping')):
            mean=group[column].mean();axis.plot(mean.index,mean.values,label=LABELS[name],color=color);axis.set(xlabel='Training epoch',ylabel=label)
    for axis in axes[:2]:axis.axhline(0,color='gray',ls='--',lw=.8)
    axes[0].legend(fontsize=7);save(fig,'neural_training')
    main=summary[(summary.horizon==30)&np.isclose(summary.cost_rate,config['cost_rate'])&summary.policy.isin(zh)].copy()
    main=main[['policy','median_growth','p10_growth','p90_loss','mean_trade_rate','mean_real_cost','review_rate']]
    main['policy']=main.policy.map(zh)
    for col in ('median_growth','p10_growth','p90_loss','mean_trade_rate','review_rate'):main[col]=main[col].map(lambda x:f'{100*x:.2f}%')
    main=main.rename(columns=dict(policy='策略',median_growth='中位实际CAGR',p10_growth='P10增长',p90_loss='P90回撤',
        mean_trade_rate='后续交易率',mean_real_cost='平均实际费用',review_rate='复核率'))
    status=screen.replace({'policy':zh,'status':{'not_established':'未建立风险改善','development_risk_improvement':'开发风险路线通过'}})
    status=status.rename(columns=dict(policy='策略',feasible_folds='可行外折数',inner_selection_feasible='全部内选可行',
        selected_training_feasible='全部训练约束通过',all_interval_conditions='全部区间通过',all_checks_passed='整体通过',status='状态'))
    chosen=trials[trials.selected][['outer_fold','policy','checkpoint','growth_difference','p10_difference','p90_difference','risk_feasible']].copy()
    chosen['policy']=chosen.policy.map(zh);chosen=chosen.rename(columns=dict(outer_fold='外折',policy='策略',checkpoint='训练轮数',
        growth_difference='中位增长差',p10_difference='P10增长差',p90_difference='P90回撤差',risk_feasible='内层可行'))
    customer=risks[['profile','policy','horizon','breaches','n','upper_bound','risk_pass','liquidity_ok','eligible']].copy();customer['policy']=customer.policy.map(zh)
    customer=customer.rename(columns=dict(profile='客户',policy='策略',horizon='期限',breaches='越界数',n='情景数',upper_bound='概率上界',risk_pass='风险筛选',liquidity_ok='流动性筛选',eligible='诊断适配'))
    body=('<h1>小型神经网络与完整路径风险约束</h1><p>只使用开发情景1–120。线性logits、16×16完整MLP和去市场特征MLP，'
        '均采用同样的目标可行域、成本门槛、复核和训练约束。每类两个固定种子，部署平均成员目标，不挑选幸运种子。</p>'
        '<p>初始统一50/35/15，目标为20/35/45、50/35/15、50/25/25的凸组合。新目标权益≤50%，'
        '保持和复核等待仍会造成持仓漂移。25%已观察实际回撤转人工，等待持有，不能保证避免进一步损失。</p>'
        '<h2>30年原费用结果</h2>'+main.to_html(index=False)+'<h2>预声明风险路线</h2>'
        '<p>相对年度Balanced，P90回撤至少改善1个百分点，中位/P10增长损失不超过0.10个百分点/年。'
        '三策略×三端点×原/双倍费用共18项调整区间，完整情景配对bootstrap20000次；须全部内选及外折所选组合训练约束均可行。'
        '区间条件于已拟合OOF模型，不包含重拟合和反复研究的不确定性。</p>'+status.to_html(index=False)+
        '<img src="figures/neural_intervals.png"><h2>训练约束与收敛诊断</h2>'
        '<p>各期限/费用均要求路径MDD的CVaR90比对应训练参照低1个百分点，最差10%年化实际对数增长尾均值最多降低0.001。'
        '这是训练经验约束，不是单年CVaR、P10 CAGR或未来保证。原始/对偶更新未满足约束时明确失败。</p>'
        '<img src="figures/neural_constraints.png"><img src="figures/neural_training.png">'
        '<p>费用、通胀、峰值和人工复核状态均在可微回测中递推；训练/部署使用相同硬费用门槛，'
        '只在已选择分支内计算梯度，不声称跨门槛可微或全局最优。交易收益门槛仍是一年均值方差代理，完整路径训练不使它成为可靠预测。</p>'
        '<h2>检查点选择</h2>'+chosen.to_html(index=False)+'<h2>客户期限风险</h2>'+customer.to_html(index=False)+
        '<p>客户按自身期限和容忍度评价；六选项调整一侧概率上界，流动性规则保持原样，不自动更改原分配。'
        '训练风险预算相对年度Balanced，不是为每位客户提供损失保证。</p>'
        '<p><a href="report.html">完整参照、期限和严重情景报告</a>；原始模型、训练市场、所有种子/检查点、'
        '对偶与梯度轨迹、内层候选和完整账本均保留。本轮开发研究，真实最终留出集未使用，不开放真实freeze/final。</p>')
    document=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>SafeInvest 神经网络路径风险研究</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#183043}'
        'img{max-width:100%}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}td,th{padding:7px;border:1px solid #ccd6dc}</style><body>'+body+'</body></html>')
    (out/'neural_report.html').write_text(document,encoding='utf-8')
    return out/'neural_report.html'
