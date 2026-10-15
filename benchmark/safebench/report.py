"""Offline, source-backed HTML report and publication-friendly PNG/SVG figures."""
import html
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from . import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'benchmark/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'equal_annual':'Equal weight / annual','equal_hold':'Equal weight / hold','bills_hold':'Bills / hold',
 'conservative_annual':'Conservative / annual','balanced_annual':'Balanced / annual','growth_annual':'Growth / annual',
 'balanced_hold':'Balanced / hold','balanced_periodic':'Balanced / 3-year','balanced_threshold':'Balanced / 5pp band',
 'minimum_variance':'Minimum variance','inverse_volatility':'Inverse volatility','mean_variance':'Mean variance',
 'cvar_constrained':'CVaR constrained','balanced_gate':'Balanced / cost gate','cost_aware':'Cost aware',
 'adaptive':'Adaptive','adaptive_review':'Adaptive + review',
 'drawdown_conservative':'Drawdown budget / 15%','drawdown_balanced':'Drawdown budget / 25%','drawdown_aggressive':'Drawdown budget / 35%',
 'drawdown_lifecycle_conservative':'DD lifecycle / 15%','drawdown_lifecycle_balanced':'DD lifecycle / 25%','drawdown_lifecycle_aggressive':'DD lifecycle / 35%',
 'budget_lifecycle_balanced':'Budget lifecycle / dynamic',
 'budget_lifecycle_balanced_zero_band':'Budget lifecycle / zero band',
 'budget_lifecycle_balanced_static':'Budget lifecycle / static gate',
 'budget_lifecycle_balanced_annual':'Budget lifecycle / annual',
 'budget_lifecycle_balanced_hold':'Budget lifecycle / hold'}

ZH_NAMES={
 'equal_annual':'等权年度再平衡','equal_hold':'等权买入持有','bills_hold':'全持有票据',
 'conservative_annual':'保守年度再平衡','balanced_annual':'均衡年度再平衡','growth_annual':'成长年度再平衡',
 'balanced_hold':'均衡买入持有','balanced_periodic':'均衡每三年再平衡','balanced_threshold':'均衡偏离5个百分点再平衡',
 'minimum_variance':'最小方差','inverse_volatility':'逆波动率','mean_variance':'均值方差',
 'cvar_constrained':'CVaR约束配置','balanced_gate':'固定均衡成本门控','cost_aware':'成本感知配置',
 'adaptive':'自适应配置','adaptive_review':'自适应加人工复核',
 'drawdown_conservative':'回撤预算（容忍度15%）','drawdown_balanced':'回撤预算（容忍度25%）',
 'drawdown_aggressive':'回撤预算（容忍度35%）',
 'drawdown_lifecycle_conservative':'生命周期回撤预算（容忍度15%）',
 'drawdown_lifecycle_balanced':'生命周期回撤预算（容忍度25%）',
 'drawdown_lifecycle_aggressive':'生命周期回撤预算（容忍度35%）',
 'budget_lifecycle_balanced':'生命周期预算动态候选',
 'budget_lifecycle_balanced_zero_band':'生命周期预算动态候选（零交易带）',
 'budget_lifecycle_balanced_static':'同初始配置静态门控参照',
 'budget_lifecycle_balanced_annual':'同初始配置年度再平衡参照',
 'budget_lifecycle_balanced_hold':'同初始配置买入持有参照','human_review':'人工复核'}


def policy_label(value, labels=ZH_NAMES):
    """Localize known policies while preserving unknown IDs instead of NaN."""
    return '未提供' if pd.isna(value) else labels.get(str(value),str(value))


def status_label(value):
    if pd.isna(value): return '未提供'
    return {'true':'是','false':'否','1':'是','0':'否'}.get(str(value).lower(),str(value))


def format_number(value, scale=1, suffix='', precision=2):
    return '未提供' if pd.isna(value) else f'{scale*value:.{precision}f}{suffix}'

def generate_report(out):
    out=Path(out); figures=out/'figures'; figures.mkdir(exist_ok=True)
    config=json.loads((out/'config.json').read_text()); manifest=json.loads((out/'manifest.json').read_text())
    summary=pd.read_csv(out/'summary.csv'); metrics=pd.read_csv(out/'scenario_metrics.csv')
    paired=pd.read_csv(out/'paired_comparisons.csv'); records=pd.read_csv(out/'decisions.csv.gz')
    main=summary[(summary.horizon==30)&np.isclose(summary.cost_rate,config['cost_rate'])].sort_values('median_growth')
    sample=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
    # An external numbered legend keeps all 28 policy labels away from the data.
    fig,ax=plt.subplots(figsize=(12,8))
    handles=[]
    for i,r in enumerate(main.itertuples()):
        point=ax.scatter(100*r.p90_loss,100*r.median_growth,s=70,
                         color=plt.cm.viridis((i+0.5)/max(1,len(main))),
                         marker='D' if bool(r.pareto) else 'o',label=f'{i+1}. {policy_label(r.policy, LABELS)}')
        ax.annotate(str(i+1),(100*r.p90_loss,100*r.median_growth),xytext=(5,3),textcoords='offset points')
        handles.append(point)
    ax.set(xlabel='90th percentile real maximum drawdown (%)',ylabel='Median real CAGR (%)',
           title='30-year risk–growth comparison · diamonds: 4-metric Pareto set')
    if handles:
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,1),fontsize=10,
                  frameon=False,title='Policy')
    ax.grid(alpha=.2)
    fig.subplots_adjust(left=.08,right=.62,bottom=.10,top=.91); save(fig,'risk_growth')
    chosen=main.policy.tolist()
    # Grow the canvas with the number of strategies so labels stay legible.
    fig_height=max(7,1.0+0.34*len(chosen))
    fig,axes=plt.subplots(1,2,figsize=(13,fig_height),sharey=True)
    for axis,metric,title in zip(axes,['median_growth','p90_loss'],['Median growth difference','P90 drawdown difference']):
        g=paired[(paired.horizon==30)&np.isclose(paired.cost_rate,config['cost_rate'])&(paired.metric==metric)].set_index('policy')
        for y,name in enumerate(chosen):
            if name not in g.index: continue
            r=g.loc[name]
            if pd.isna(r.lower95) or pd.isna(r.upper95) or pd.isna(r.difference): continue
            axis.plot([100*r.lower95,100*r.upper95],[y,y],color='#287c8e',lw=2)
            axis.scatter(100*r.difference,y,color='#173e5b',s=24)
        axis.axvline(0,color='gray',ls='--'); axis.set_title(title); axis.set_xlabel('Percentage points vs equal weight')
        axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(chosen)),[policy_label(p,LABELS) for p in chosen])
    fig.suptitle('Paired scenario bootstrap · pointwise 95% intervals\nExploratory OOF comparisons; fitted-model uncertainty excluded',fontsize=11)
    fig.tight_layout(); save(fig,'paired_intervals')
    selected=[p for p in ['equal_annual','balanced_annual','balanced_hold','budget_lifecycle_balanced',
                          'budget_lifecycle_balanced_zero_band','budget_lifecycle_balanced_static',
                          'budget_lifecycle_balanced_annual','budget_lifecycle_balanced_hold',
                          'cost_aware','adaptive','adaptive_review'] if p in set(main.policy)]
    styles={name:dict(color=plt.cm.viridis((i+0.5)/max(1,len(selected))),
                      linestyle=['-','--','-.',':'][i%4],marker=['o','s','^','D'][i%4])
            for i,name in enumerate(selected)}
    fig,axes=plt.subplots(1,2,figsize=(13,6.5))
    for name in selected:
        g=summary[(summary.policy==name)&(summary.horizon==30)].sort_values('cost_rate')
        axes[0].plot(100*g.cost_rate,100*g.median_growth,label=policy_label(name,LABELS),**styles[name])
        axes[1].plot(100*g.cost_rate,100*g.mean_trade_rate,label=policy_label(name,LABELS),**styles[name])
    axes[0].set(ylabel='Median real CAGR (%)',xlabel='Cost per transfer (%)',title='Cost sensitivity · frozen parameters')
    axes[1].set(ylabel='Post-initial reviews with trades (%)',xlabel='Cost per transfer (%)',title='Trading efficiency')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,0),ncol=3,fontsize=10,frameon=False)
    fig.tight_layout(rect=(0,.24,1,1)); save(fig,'cost_sensitivity')
    # Choose severe scenario by the official reference only, before comparing strategies on that scenario.
    severe=int(sample[sample.policy=='equal_annual'].sort_values(['real_max_drawdown','scenario_id'],ascending=[False,True]).iloc[0].scenario_id)
    severe_rows=sample[sample.scenario_id==severe].copy(); severe_rows.to_csv(out/'severe_scenario.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,6))
    severe_records=records[(records.scenario_id==severe)&np.isclose(records.cost_rate,config['cost_rate'])]
    # Legacy records have no horizon column; newer horizon-specific paths must not mix.
    if 'horizon' in severe_records.columns:
        severe_records=severe_records[severe_records.horizon==30]
    for name in selected:
        g=severe_records[severe_records.policy==name].sort_values('period')
        if g.empty: continue
        ax.plot([0]+g.period.tolist(),[config['initial_wealth']]+g.real_wealth.tolist(),
                label=policy_label(name,LABELS),**styles[name])
        alert=g[g.action=='human_review']
        if len(alert): ax.scatter(alert.period,alert.real_wealth,marker='X',s=90,color=styles[name]['color'])
    ax.axhline(config['initial_wealth'],color='gray',ls='--',lw=.8)
    ax.set(xlabel='Annual period',ylabel=f'Real wealth (initial = {config["initial_wealth"]:g})',
           title=f'Scenario {severe}: worst reference drawdown in this evaluation')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=10,frameon=False)
    fig.subplots_adjust(left=.09,right=.64,bottom=.12,top=.90); save(fig,'severe_path')
    fig,ax=plt.subplots(figsize=(12,6))
    for name in selected:
        g=summary[(summary.policy==name)&np.isclose(summary.cost_rate,config['cost_rate'])].sort_values('horizon')
        ax.plot(g.horizon,100*g.p90_loss,label=policy_label(name,LABELS),**styles[name])
    ax.set(xlabel='Horizon (years)',ylabel='P90 real maximum drawdown (%)',title='Risk depends on the investment horizon')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=10,frameon=False)
    fig.subplots_adjust(left=.09,right=.64,bottom=.12,top=.90); save(fig,'horizon_risk')
    columns=['policy','median_growth','p10_growth','median_loss','p90_loss','median_volatility','mean_trade_rate','mean_real_cost','review_rate','pareto']
    display=main.reindex(columns=columns).sort_values('median_growth',ascending=False).copy()
    for col in columns[1:]:
        if col not in ('pareto','mean_real_cost'):
            display[col]=display[col].map(lambda x:format_number(x,100,'%'))
    display.mean_real_cost=display.mean_real_cost.map(lambda x:format_number(x,precision=3))
    # Keep stable machine-facing IDs alongside localized presentation.
    display.insert(1,'policy_id',display['policy'])
    display['policy']=display['policy'].map(policy_label)
    display['pareto']=display['pareto'].map(status_label)
    display=display.rename(columns={'policy':'策略','policy_id':'策略ID','median_growth':'中位实际CAGR','p10_growth':'P10增长',
        'median_loss':'中位回撤','p90_loss':'P90回撤','median_volatility':'中位普通波动',
        'mean_trade_rate':'后续交易率','mean_real_cost':'平均实际费用','review_rate':'复核率','pareto':'Pareto集合'})
    table=display.to_html(index=False,escape=True)
    title='开发集交叉验证' if manifest['phase']=='development' else '冻结后的独立最终评价'
    parts=[f'<h1>SafeInvest · {title}</h1>',
      f'<p>{len(manifest["scenario_ids"])} 条独立情景；转移成本率 {100*config["cost_rate"]:.2f}%；结果已扣除费用和通胀。'
      '增长为实际 CAGR，损失为包含每次扣费时点的实际最大回撤，数值越大损失越严重。图表保留英文标签，便于比赛报告引用。</p>',
      '<p class="caveat">开发结果是探索性的折外估计，不是最终测试证据。交叉验证不能消除重复研究的选择偏差。'
      'Bootstrap 区间条件于已拟合策略，未包含重拟合不确定性，且外层训练折存在重叠。'
      '多候选调整区间见 paired_comparisons.csv。</p>' if manifest['phase']=='development' else
      '<p class="caveat">最终留出集已使用。请报告全部预声明比较，不再据此调参。</p>',
      '<h2>30 年期限比较</h2><p>交易率不包含建仓。费用是累计实际费用，不是终值机会损失。'
      '部分行是无收益门槛的诊断参照，不能直接当成合规的顾问建议策略。</p>'+table,
      '<h2>增长与下行风险</h2><img src="figures/risk_growth.png" alt="增长与P90回撤比较">',
      '<h2>配对比较的不确定性</h2><img src="figures/paired_intervals.png" alt="配对Bootstrap区间">',
      '<h2>成本敏感性</h2><img src="figures/cost_sensitivity.png" alt="费用及交易率敏感性">',
      f'<h2>严重情景 {severe}</h2><p>仅按官方等权最大回撤选择，不代表总体最坏情景。'
      'X 表示人工复核所在年份；此后资产继续漂移，没有虚构顾问干预。</p>'
      '<img src="figures/severe_path.png" alt="严重情景的实际财富路径">',
      '<h2>投资期限敏感性</h2><img src="figures/horizon_risk.png" alt="不同投资期限的回撤">']
    evidence_files=[('incremental_comparisons.csv','同敞口参照的配对增量'),
                    ('incremental_screen.csv','预声明增量与服务筛选')]
    for filename,heading in evidence_files:
        path=out/filename
        if not path.exists(): continue
        try:
            evidence=pd.read_csv(path)
        except pd.errors.EmptyDataError:
            evidence=pd.DataFrame()
        parts.append(f'<h2>{heading}</h2>')
        if evidence.empty:
            parts.append('<p>文件没有可展示的记录，不能据此判断增量价值。</p>')
            continue
        if filename=='incremental_comparisons.csv':
            endpoints={'median_growth':'中位增长差及族调整区间（百分点）',
                       'p10_growth':'P10增长差及族调整区间（百分点）',
                       'p90_loss':'P90回撤差及族调整区间（百分点）'}
            evidence=evidence[evidence.metric.isin(endpoints)].copy()
            evidence['endpoint']=evidence.metric.map(endpoints)
            evidence['estimate']=evidence.apply(lambda r:
                f'{format_number(r.get("difference"),100)} '
                f'[{format_number(r.get("family_lower"),100)}, {format_number(r.get("family_upper"),100)}]',axis=1)
            keys=[c for c in ['policy','reference','horizon','cost_rate','n'] if c in evidence.columns]
            evidence=evidence.pivot_table(index=keys,columns='endpoint',values='estimate',aggfunc='first').reset_index()
            evidence.columns.name=None
            if 'cost_rate' in evidence.columns:
                evidence['cost_rate']=evidence.cost_rate.map(lambda x:format_number(x,100,'%'))
            parts.append('<p>差值为动态候选减参照；增长差为正较好、回撤差为负较好。'
                         '区间按预声明参照与期限族调整；这些条件于已拟合模型的探索性区间不含重拟合不确定性。</p>')
        else:
            keep=['policy','reference','horizon','incremental_value_established',
                  'breach_upper','review_rate','service_screen_pass','decision']
            evidence=evidence[[c for c in keep if c in evidence.columns]].copy()
            for col in ['risk_reduction_pass','median_growth_noninferior','p10_growth_noninferior',
                        'double_cost_checked','double_cost_pass','incremental_value_established','service_screen_pass']:
                if col in evidence.columns: evidence[col]=evidence[col].map(status_label)
            for col in ['breach_upper','review_rate']:
                if col in evidence.columns: evidence[col]=evidence[col].map(lambda x:format_number(x,100,'%'))
            if 'decision' in evidence.columns:
                evidence['decision']=evidence.decision.replace({'research_screen_pass':'研究筛选通过',
                                                                               'not_established':'尚未建立增量价值'})
            parts.append('<p class="caveat">预声明增量筛选未通过时，不能宣称动态预算相对相应同敞口参照具有增量价值；'
                         '单个端点改善、Pareto 入选或零交易带换手下降均不足以替代完整筛选。'
                         '研究筛选通过也不等于客户服务适配通过，更不表示可绕过交易收益门槛。'
                         '未检查的双倍成本结果不视为通过。</p>')
        for col in ['policy','reference']:
            if col in evidence.columns:
                index=evidence.columns.get_loc(col)
                evidence.insert(index+1,col+'_id',evidence[col])
                evidence[col]=evidence[col].map(policy_label)
        evidence=evidence.rename(columns={'policy':'候选','policy_id':'候选ID','reference':'参照','reference_id':'参照ID',
            'horizon':'期限（年）','cost_rate':'转移成本率','n':'配对情景数','risk_reduction_pass':'风险降低',
            'median_growth_noninferior':'中位增长非劣','p10_growth_noninferior':'P10增长非劣',
            'risk_consistent_folds':'风险改善折数','double_cost_checked':'已查双倍成本','double_cost_pass':'双倍成本通过',
            'incremental_value_established':'增量价值筛选通过','breach_upper':'越界概率上界',
            'review_rate':'复核率','service_screen_pass':'服务筛选通过','decision':'结论'})
        parts.append(evidence.fillna('未提供').to_html(index=False,escape=True))
    if (out/'suitability.csv').exists():
        suitability=pd.read_csv(out/'suitability.csv')
        if 'portfolio' in suitability.columns:
            suitability['portfolio']=suitability['portfolio'].replace({'conservative':'保守','balanced':'均衡','growth':'成长','human_review':'人工复核'}).map(policy_label)
        if 'policy' in suitability.columns:
            suitability.insert(suitability.columns.get_loc('policy')+1,'policy_id',suitability['policy'])
            suitability['policy']=suitability['policy'].map(policy_label)
        if 'liquidity' in suitability.columns:
            suitability['liquidity']=suitability['liquidity'].replace({'high':'高','medium':'中','low':'低'})
        if 'explanation' in suitability.columns:
            suitability['explanation']=suitability['explanation'].replace({
                'Choose the eligible portfolio with highest median real growth; loss tolerance is not guaranteed.':'选择符合筛选条件且中位实际增长最高的组合；不保证损失不超出容忍度。',
                'No portfolio passes the conservative development risk screen and liquidity/horizon rules.':'没有组合同时通过开发风险筛选及流动性/期限规则。',
                'Choose the eligible policy with highest median real growth; loss tolerance is not guaranteed.':'选择符合筛选条件且中位实际增长最高的策略；不保证损失不超出容忍度。',
                'No policy passes the conservative development risk screen and liquidity/horizon rules.':'没有策略同时通过开发风险筛选及流动性/期限规则。'})
        if 'loss_tolerance' in suitability.columns:
            suitability['loss_tolerance']=suitability['loss_tolerance'].map(lambda x:format_number(x,100,'%',0))
        suitability=suitability.rename(columns={'profile_id':'客户','horizon':'期限（年）','loss_tolerance':'损失容忍度',
            'liquidity':'流动性需要','portfolio':'组合/动作','policy':'策略','policy_id':'策略ID','explanation':'说明',
            'final_breach_probability':'最终越界概率','final_median_growth':'最终中位增长'})
        parts+=['<h2>客户入门适配示例</h2><p>固定组合与生命周期预算动态候选（如本次运行包含）均须按客户期限接受风险、'
                '流动性和服务筛选；纳入候选不等于通过或获推荐。高流动性规则及成长组合至少15年的限制仍适用。'
                '风险界应覆盖本次实际候选集合，不能沿用仅三种固定组合的校正。'
                '容忍度是客户回答，不是保证；20%越界概率筛选是研究假设，研究增量价值也不能替代客户适配。</p>',
                suitability.fillna('未提供').to_html(index=False,escape=True)]
    method_link=html.escape(os.path.relpath(ROOT/'docs/METHODOLOGY.md',out),quote=True)
    protocol_link=html.escape(os.path.relpath(ROOT/'docs/PROTOCOL.md',out),quote=True)
    parts+=['<h2>核验与解释</h2><p>逐情景重分析见 scenario_metrics.csv，年度决策见 decisions.csv.gz，'
            '来源、代码、环境和种子见 manifest.json。'
            f'<a href="{method_link}">数学定义</a> · <a href="{protocol_link}">实验协议</a>。'
            '合成情景支持生成器内的比较，不证明实盘效果。</p>',
            '<p>数据：提供的 SafeInvest 合成数据，基于 Jordà–Schularick–Taylor Macrohistory Database Release 6，'
            '许可 CC BY-NC-SA 4.0。图表和表格为派生分析。AI 使用：OpenAI Codex 协助编写代码和文档；团队须在提交前核查。</p>']
    body='\n'.join(parts)
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SafeInvest 基准报告</title><style>body{font:16px/1.5 system-ui;margin:32px auto;max-width:1250px;padding:0 20px;color:#183043}h1,h2{line-height:1.2}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}td,th{padding:7px;border:1px solid #ccd6dc;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}.caveat{background:#fff2ce;padding:16px}</style><body>'+body+'</body></html>'
    (out/'report.html').write_text(document)
    (out/'comparison_table.csv').write_text(display.to_csv(index=False))
    return out/'report.html'
