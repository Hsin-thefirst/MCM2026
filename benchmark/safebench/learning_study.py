"""Preregistered nested study, grouped inference and Chinese reporting for fitted Q policies."""
from dataclasses import asdict
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from safeinvest.contracts import SimulationConfig, validate_weights
from .learning_core import LEARNING_NAMES, FEATURE_NAMES, save_training_data


def validate_learning_config(config):
    if config.get('study_kind') != 'counterfactual_fqi':
        raise ValueError('Wrong learning study kind')
    penalties = config['learning_penalties']
    if not penalties or len(set(penalties)) != len(penalties) or any(not np.isfinite(v) or v < 0 for v in penalties):
        raise ValueError('Invalid learning penalties')
    if len(config['learning_targets']) != 3:
        raise ValueError('Exactly three learning targets required')
    for target in config['learning_targets']:
        validate_weights(target)
    for key in ('learning_teacher_depth', 'learning_student_depth', 'learning_min_leaf', 'learning_split_candidates'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError('Invalid tree configuration')
    for key in ('learning_advantage_scale', 'learning_growth_margin', 'learning_p10_margin', 'learning_p90_selection_margin'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Invalid learning scale or margin')
    if not 0 < config['review_drawdown'] < config['learning_loss_tolerance'] < 1:
        raise ValueError('Invalid review/loss thresholds')
    if not 0 < config['cost_rate'] < .5 or not np.isfinite(config['learning_buffer']) or config['learning_buffer'] < 0:
        raise ValueError('Invalid learning cost/buffer')
    if not any(np.isclose(v, 2 * config['cost_rate']) for v in config['sensitivity_costs']):
        raise ValueError('Doubled cost must be evaluated')
    if not {5, 6, 15, 25, 30} <= set(config['horizons']):
        raise ValueError('Customer horizons must be simulated separately')


def evaluate_learning(scenarios, model, params, config, names=None, cost=None, horizons=None, fold=0, keep_records=True):
    from .policies import make_policy
    from .engine import simulate
    from .metrics import compute_metrics
    names = names or config['evaluation_policy_names']
    cost = config['cost_rate'] if cost is None else cost
    horizons = horizons or config['horizons']
    output, records = [], []
    if any(s.scenario_id in model['train_ids'] for s in scenarios):
        raise ValueError('Evaluation scenario overlaps fitted training scenarios')
    for name in names:
        selected = params.get(name, config['tuning_grid'][0])
        for scenario in scenarios:
            for horizon in horizons:
                policy = make_policy(name, model, selected, config, horizon=horizon)
                result = simulate(scenario, policy, policy.initial_target,
                    SimulationConfig(horizon, config['initial_wealth'], cost))
                output.append(dict(**compute_metrics(result), fold=fold,
                    inference_seconds=getattr(policy,'inference_seconds',0.),inference_calls=getattr(policy,'inference_calls',0)))
                if keep_records:
                    for record in result.records:
                        row = asdict(record)
                        for field in ('target_weights', 'pre_trade_holdings', 'post_trade_holdings', 'end_holdings'):
                            values = row.pop(field)
                            row.update({field + '_' + asset: v for asset, v in zip(('equity', 'bond', 'bill'), values)})
                        row.update(scenario_id=scenario.scenario_id, policy=name, cost_rate=cost, fold=fold,
                                   evaluation_horizon=horizon, action=record.action.value,
                                   reason_codes=json.dumps(record.reason_codes))
                        records.append(row)
    return pd.DataFrame(output), pd.DataFrame(records)


def select_learning_trials(scores, reference, config):
    rows = []
    ref = reference.set_index('scenario_id').sort_index()
    if ref.index.duplicated().any():
        raise ValueError('Unpaired learning reference')
    for name in LEARNING_NAMES:
        for trial, penalty in enumerate(config['learning_penalties']):
            group = scores[(scores.policy == name) & (scores.trial == trial)].set_index('scenario_id').sort_index()
            if not group.index.equals(ref.index) or group.index.duplicated().any():
                raise ValueError('Unpaired learning candidate')
            growth = float(group.real_cagr.median() - ref.real_cagr.median())
            p10 = float(group.real_cagr.quantile(.1) - ref.real_cagr.quantile(.1))
            p90 = float(group.real_max_drawdown.quantile(.9))
            ref_p90 = float(ref.real_max_drawdown.quantile(.9))
            violation = max(0., (-growth - config['learning_growth_margin']) / config['learning_growth_margin'],
                (-p10 - config['learning_p10_margin']) / config['learning_p10_margin'],
                (p90 - ref_p90 - config['learning_p90_selection_margin']) / config['learning_p90_selection_margin'])
            rows.append(dict(policy=name, trial=trial, penalty=penalty, growth_difference=growth,
                p10_difference=p10, p90_loss=p90, reference_p90_loss=ref_p90, p90_difference=p90-ref_p90,
                breach_probability=float((group.real_max_drawdown > config['learning_loss_tolerance']).mean()),
                median_growth=float(group.real_cagr.median()), risk_feasible=violation <= 1e-10,
                normalized_violation=violation, selected=False, selection_status='not_selected'))
    selected = {}
    for name in ('fqi_student', 'fqi_portfolio'):
        options = [r for r in rows if r['policy'] == name]
        feasible = [r for r in options if r['risk_feasible']]
        chosen = min(feasible or options, key=lambda r: (
            0 if feasible else r['normalized_violation'], r['p90_loss'], r['breach_probability'], -r['median_growth'], r['trial']))
        selected[name] = chosen['trial']
    selected['fqi_teacher'] = selected['fqi_student']
    params = {}
    for row in rows:
        if row['trial'] != selected[row['policy']]:
            continue
        row['selected'] = True
        row['selection_status'] = 'feasible' if row['risk_feasible'] else 'infeasible_diagnostic'
        params[row['policy']] = dict(penalty=row['penalty'], selection_trial=row['trial'], selection_status=row['selection_status'])
    return params, rows


def tune_learning(scenarios, config, seed, trace_path):
    from .data import folds
    from .policies import fit_model
    from .experiment import dump
    validate_learning_config(config)
    trace_path = Path(trace_path) if trace_path is not None else None
    if trace_path is not None and trace_path.exists():
        raise FileExistsError(trace_path)
    by_id = {s.scenario_id: s for s in scenarios}
    collected, references, audits = [], [], []
    for inner, validation_ids in enumerate(folds(by_id, config['inner_folds'], seed)):
        train = [s for sid, s in by_id.items() if sid not in validation_ids]
        valid = [by_id[sid] for sid in validation_ids]
        model = fit_model(train, config)
        audits.append(dict(inner_fold=inner, train_ids=model['train_ids'], validation_ids=validation_ids))
        if trace_path is not None:
            model_dir = trace_path.parent.parent / 'tuning_models'; model_dir.mkdir(parents=True, exist_ok=True)
            data_dir = trace_path.parent.parent / 'training_data'; data_dir.mkdir(exist_ok=True)
            stem = trace_path.name.split('.')[0] + f'_inner_{inner}'
            dump(model_dir / (stem + '.json'), model)
            save_training_data(train, config, model['fqi'], data_dir / (stem + '.npz'))
        ref, _ = evaluate_learning(valid, model, {}, config, names=[config['learning_reference']], horizons=[30],
                                   fold=inner, keep_records=False)
        ref['trial'] = -1; references.append(ref)
        for trial, penalty in enumerate(config['learning_penalties']):
            metrics, _ = evaluate_learning(valid, model, {n: dict(penalty=penalty) for n in LEARNING_NAMES},
                config, names=LEARNING_NAMES, horizons=[30], fold=inner, keep_records=False)
            metrics['trial'] = trial; collected.append(metrics)
    scores, reference = pd.concat(collected, ignore_index=True), pd.concat(references, ignore_index=True)
    chosen, rows = select_learning_trials(scores, reference, config)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([scores, reference], ignore_index=True).to_csv(trace_path, index=False,
            compression={'method': 'gzip', 'mtime': 0})
    return chosen, rows, audits


def learning_intervals(metrics, config, policy_names=LEARNING_NAMES, prefix='learning'):
    rng = np.random.default_rng(config['seed'])
    samples = config[prefix+'_bootstrap']
    family = len(policy_names) * 3 * 2
    tail = .05 / (2 * family)
    rows = []
    for cost in (config['cost_rate'], 2 * config['cost_rate']):
        block = metrics[(metrics.horizon == 30) & np.isclose(metrics.cost_rate, cost)]
        reference = block[block.policy == config[prefix+'_reference']].set_index('scenario_id').sort_index()
        indices = rng.integers(0, len(reference), (samples, len(reference)))
        for name in policy_names:
            candidate = block[block.policy == name].set_index('scenario_id').sort_index()
            if not candidate.index.equals(reference.index) or candidate.index.duplicated().any():
                raise ValueError('Unpaired learning intervals')
            for column, q, label, threshold, side in (
                ('real_cagr', .5, 'median_growth', -config[prefix+'_growth_margin'], 'lower'),
                ('real_cagr', .1, 'p10_growth', -config[prefix+'_p10_margin'], 'lower'),
                ('real_max_drawdown', .9, 'p90_loss', -config[prefix+'_p90_improvement'], 'upper')):
                a, b = candidate[column].to_numpy(), reference[column].to_numpy()
                draws = np.quantile(a[indices], q, axis=1) - np.quantile(b[indices], q, axis=1)
                low, high = float(np.quantile(draws, tail)), float(np.quantile(draws, 1-tail))
                rows.append(dict(policy=name, reference=config[prefix+'_reference'], horizon=30, cost_rate=cost,
                    metric=label, n=len(reference), difference=float(np.quantile(a,q)-np.quantile(b,q)),
                    lower95=float(np.quantile(draws,.025)), upper95=float(np.quantile(draws,.975)),
                    family_lower=low, family_upper=high, family_comparisons=family, bootstrap=samples,
                    threshold=threshold, bound=side, passed=low >= threshold if side == 'lower' else high <= threshold))
    return pd.DataFrame(rows)


def write_learning_analysis(out, config):
    from . import ROOT
    from .metrics import binomial_upper, paired_comparisons
    out = Path(out)
    metrics, trials = pd.read_csv(out/'scenario_metrics.csv'), pd.read_csv(out/'tuning_trials.csv')
    intervals = learning_intervals(metrics, config)
    intervals.to_csv(out/'learning_intervals.csv', index=False)
    screen = []
    for name in LEARNING_NAMES:
        chosen = trials[(trials.policy == name) & trials.selected]
        feasible = len(chosen) == config['outer_folds'] and chosen.risk_feasible.all()
        passed = bool(feasible and intervals[intervals.policy==name].passed.all())
        screen.append(dict(policy=name, feasible_folds=int(chosen.risk_feasible.sum()),
            inner_selection_feasible=bool(feasible), all_interval_conditions=bool(intervals[intervals.policy==name].passed.all()),
            all_checks_passed=passed, status='development_risk_improvement' if passed else 'not_established'))
    pd.DataFrame(screen).to_csv(out/'learning_screen.csv', index=False)
    profiles = json.loads((ROOT/'configs/profiles.json').read_text(encoding='utf-8'))
    risks = []
    confidence = 1 - (1 - config['suitability_confidence']) / (3 + len(LEARNING_NAMES))
    for profile in profiles:
        for name in LEARNING_NAMES:
            group = metrics[(metrics.policy==name)&(metrics.horizon==profile['horizon'])&np.isclose(metrics.cost_rate,config['cost_rate'])]
            if group.empty or group.scenario_id.duplicated().any():
                raise ValueError('Missing or duplicated customer learning horizon')
            breaches = int((group.real_max_drawdown > profile['loss_tolerance']).sum())
            upper = binomial_upper(breaches, len(group), confidence)
            liquid = profile['liquidity'] != 'high'
            risks.append(dict(profile=profile['profile_id'], policy=name, horizon=profile['horizon'], tolerance=profile['loss_tolerance'],
                n=len(group), breaches=breaches, breach_probability=breaches/len(group), confidence=confidence, upper_bound=upper,
                liquidity_ok=liquid, risk_pass=upper <= config['suitability_max_breach_probability'],
                eligible=bool(liquid and upper <= config['suitability_max_breach_probability']),
                median_growth=float(group.real_cagr.median()), diagnostic_only=True,
                training_loss_tolerance=config['learning_loss_tolerance']))
    pd.DataFrame(risks).to_csv(out/'learning_suitability.csv', index=False)
    subset = metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])&
                     metrics.policy.isin(['balanced_annual_review',*LEARNING_NAMES])]
    paired_comparisons(subset,config['seed'],config['bootstrap'],baseline='balanced_annual_review').to_csv(
        out/'learning_review_comparisons.csv',index=False)
    fitted = json.loads((out/'fitted_models.json').read_text())
    training, rules = [], []
    for fold in fitted:
        for feature_set, penalties in fold['model']['fqi'].items():
            for penalty, model in penalties.items():
                for trace in model['trace']:
                    training.append(dict(fold=fold['fold'],feature_set=feature_set,penalty=float(penalty),**trace))
        penalty = str(fold['params']['fqi_student']['penalty'])
        tree = fold['model']['fqi']['full'][penalty]['models']['29']['student']
        def visit(node, conditions):
            if 'feature' not in node:
                rules.append(dict(fold=fold['fold'],remaining=29,conditions=conditions,encoded_q=node['value'],training_rows=node['n']))
            else:
                label = FEATURE_NAMES[node['feature']]
                visit(node['left'],conditions+[f"{label} <= {node['threshold']:.6g}"])
                visit(node['right'],conditions+[f"{label} > {node['threshold']:.6g}"])
        visit(tree,[])
    pd.DataFrame(training).to_csv(out/'learning_training.csv',index=False)
    (out/'learning_rules.json').write_text(json.dumps(rules,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def generate_learning_report(out, config):
    from .report import plt, LABELS
    out = Path(out)
    intervals = pd.read_csv(out/'learning_intervals.csv')
    screen = pd.read_csv(out/'learning_screen.csv')
    summary = pd.read_csv(out/'summary.csv')
    training = pd.read_csv(out/'learning_training.csv')
    risks = pd.read_csv(out/'learning_suitability.csv')
    trials = pd.read_csv(out/'tuning_trials.csv')
    zh = {'fqi_student':'完整特征学生树','fqi_teacher':'配套教师树','fqi_portfolio':'仅持仓回撤学生树',
          'balanced_annual':'年度Balanced','balanced_annual_review':'年度Balanced加相同复核','equal_annual':'官方等权'}
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg','pdf'):
            fig.savefig(out/'figures'/(name+'.'+extension),dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for axis, endpoint, label in zip(axes,('median_growth','p10_growth','p90_loss'),
        ('Median real CAGR difference\n(pp / year)','P10 real CAGR difference\n(pp / year)','P90 drawdown difference\n(pp)')):
        group=intervals[(intervals.metric==endpoint)&np.isclose(intervals.cost_rate,config['cost_rate'])].set_index('policy')
        for y,name in enumerate(LEARNING_NAMES):
            row=group.loc[name]
            axis.plot(100*np.array([row.family_lower,row.family_upper]),[y,y],color='#287c8e',lw=2)
            axis.scatter(100*row.difference,y,color='#173e5b')
        axis.axvline(100*group.iloc[0].threshold,color='#ae5b32',ls='--')
        axis.axvline(0,color='gray',ls=':',lw=.8)
        axis.set_xlabel(label);axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(3),[LABELS[n] for n in LEARNING_NAMES]);save(fig,'learning_intervals')
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for family,color in (('full','#287c8e'),('portfolio','#ae5b32')):
        group=training[training.feature_set==family].groupby('remaining')
        axes[0].plot(group.training_action_agreement.mean().index,100*group.training_action_agreement.mean(),label=family,color=color)
        axes[1].plot(group.distillation_rmse.mean().index,group.distillation_rmse.mean(),label=family,color=color)
    axes[0].set(xlabel='Remaining annual periods',ylabel='Training action agreement (%)',ylim=(0,105))
    axes[1].set(xlabel='Remaining annual periods',ylabel='Q-value distillation RMSE')
    axes[0].legend();axes[1].legend();save(fig,'learning_distillation')
    records=pd.read_csv(out/'decisions.csv.gz')
    block=records[(records.evaluation_horizon==30)&np.isclose(records.cost_rate,config['cost_rate'])&(records.period>1)]
    fig,axis=plt.subplots(figsize=(9,4))
    categories=['keep','rebalance','change_target','human_review']
    left=np.zeros(3)
    for action,color in zip(categories,('#8eb8c4','#287c8e','#ae5b32','#a25378')):
        values=np.array([100*(block[block.policy==name].action==action).mean() for name in LEARNING_NAMES])
        axis.barh(range(3),values,left=left,label=action,color=color);left+=values
    axis.set_yticks(range(3),[LABELS[n] for n in LEARNING_NAMES]);axis.set_xlabel('Post-initial review actions (%)')
    axis.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=4,fontsize=8);save(fig,'learning_actions')
    main=summary[(summary.horizon==30)&np.isclose(summary.cost_rate,config['cost_rate'])&
        summary.policy.isin(['equal_annual','balanced_annual','balanced_annual_review',*LEARNING_NAMES])].copy()
    main=main[['policy','median_growth','p10_growth','p90_loss','mean_trade_rate','mean_real_cost','review_rate']]
    main['policy']=main.policy.map(zh)
    for col in ('median_growth','p10_growth','p90_loss','mean_trade_rate','review_rate'):
        main[col]=main[col].map(lambda v:f'{100*v:.2f}%')
    main=main.rename(columns=dict(policy='策略',median_growth='中位实际CAGR',p10_growth='P10增长',p90_loss='P90回撤',
                                  mean_trade_rate='后续交易率',mean_real_cost='平均实际费用',review_rate='复核率'))
    status=screen.replace({'policy':zh,'status':{'not_established':'未确立风险改善','development_risk_improvement':'开发风险路线通过'}})
    status=status.rename(columns=dict(policy='策略',feasible_folds='可行外折数',inner_selection_feasible='全部内选可行',
        all_interval_conditions='全部区间通过',all_checks_passed='整体通过',status='状态'))
    selected=trials[trials.selected][['outer_fold','policy','penalty','growth_difference','p10_difference','p90_difference','risk_feasible']].copy()
    selected['policy']=selected.policy.map(zh)
    selected=selected.rename(columns=dict(outer_fold='外折',policy='策略',penalty='风险惩罚',growth_difference='中位增长差',
        p10_difference='P10增长差',p90_difference='P90回撤差',risk_feasible='点估计约束可行'))
    customer=risks[['profile','policy','horizon','breaches','n','upper_bound','risk_pass','liquidity_ok','eligible']].copy()
    customer['policy']=customer.policy.map(zh)
    customer=customer.rename(columns=dict(profile='客户',policy='策略',horizon='期限',breaches='越界数',n='情景数',
        upper_bound='概率上界',risk_pass='风险筛选',liquidity_ok='流动性筛选',eligible='诊断适配'))
    rules=json.loads((out/'learning_rules.json').read_text())
    rule_text='\n'.join(' AND '.join(r['conditions'])+' -> encoded Q = '+str([round(v,5) for v in r['encoded_q']]) for r in rules if r['fold']==0)
    body=('<h1>反事实离线强化学习与可解释学生树</h1><p>仅使用情景1–120。四个动作：保持、目标30/30/40、'
        '50/35/15、60/25/15；初始统一50/35/15。完整特征教师深度4，学生深度2；去除滞后市场特征为消融。'
        '教师沿用主学生所选惩罚，三策略全部报告，不根据外折结果追加搜索。</p>'
        '<p>学习奖励为实际对数增长减首次越过40%实际最大回撤的惩罚。25%已观察实际回撤进入持续人工复核，'
        '等待持有；这不保证避免后续损失。按客户剩余期限分别模拟，短期限结果不是30年结果前缀。</p>'
        '<h2>30年原费用结果</h2>'+main.to_html(index=False)+'<h2>预声明风险路线</h2>'
        '<p>相对年度Balanced，P90回撤须至少改善1个百分点，中位/P10年增长损失均不超过0.10个百分点；'
        '原/双倍费用下18项调整区间均通过且全部内选可行才算通过。区间按完整情景配对，条件于已拟合OOF模型，'
        '不包含重拟合和重复开发的不确定性。</p>'+status.to_html(index=False)+
        '<img src="figures/learning_intervals.png"><h2>蒸馏与动作</h2>'
        '<p>下图是一组教师到学生的训练内压缩诊断，不代表折外表现；折外财务结果见上表。'
        'Q值是风险惩罚后的价值代理，不是保本承诺。当前费用由解析公式扣除一次，交易仍须覆盖费用和缓冲。</p>'
        '<img src="figures/learning_distillation.png"><img src="figures/learning_actions.png">'
        '<h2>内层选择</h2>'+selected.to_html(index=False)+'<h2>客户期限诊断</h2>'
        '<p>三策略均按40%训练风险目标学习；各客户仍按自身容忍度核验，未声称完成个性化风险训练。'
        '一侧概率上界对原三组合及三学习策略共六选项调整；不自动修改原客户分配。</p>'+customer.to_html(index=False)+
        '<h2>学生树示例</h2><p>第一个外折、剩余29年的树；叶输出是动作价值，之后还需按当前持仓与费用门槛选动作。'
        '所有期限和折的树均保存在fitted_models.json，规则示例见learning_rules.json。</p><pre>'+html.escape(rule_text)+
        '</pre><p><a href="report.html">完整参照、期限与严重情景报告</a>；原始指标、完整账本、内层候选、'
        '反事实训练数据与模型保存在本目录。本轮仅开发研究，不开放真实冻结/最终评价；最终留出集未使用。</p>')
    document=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>SafeInvest 离线强化学习研究</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#183043}'
        'img{max-width:100%}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}td,th{padding:7px;border:1px solid #ccd6dc}'
        'pre{overflow:auto;background:#eef4f6;padding:16px}</style><body>'+body+'</body></html>')
    (out/'learning_report.html').write_text(document,encoding='utf-8')
    return out/'learning_report.html'
