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
 'drawdown_lifecycle_conservative':'DD lifecycle / 15%','drawdown_lifecycle_balanced':'DD lifecycle / 25%','drawdown_lifecycle_aggressive':'DD lifecycle / 35%'}

def generate_report(out):
    out=Path(out); figures=out/'figures'; figures.mkdir(exist_ok=True)
    config=json.loads((out/'config.json').read_text()); manifest=json.loads((out/'manifest.json').read_text())
    summary=pd.read_csv(out/'summary.csv'); metrics=pd.read_csv(out/'scenario_metrics.csv')
    paired=pd.read_csv(out/'paired_comparisons.csv'); records=pd.read_csv(out/'decisions.csv.gz')
    main=summary[(summary.horizon==30)&np.isclose(summary.cost_rate,config['cost_rate'])].sort_values('median_growth')
    sample=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for i,r in enumerate(main.itertuples()):
        ax.scatter(100*r.p90_loss,100*r.median_growth,s=70,marker='D' if r.pareto else 'o')
        ax.annotate(str(i+1),(100*r.p90_loss,100*r.median_growth),xytext=(5,3),textcoords='offset points')
    ax.set(xlabel='90th percentile real maximum drawdown (%)',ylabel='Median real CAGR (%)',
           title='30-year risk–growth comparison · diamonds: 4-metric Pareto set')
    ax.text(1.03,.98,'\n'.join(f'{i+1}. {LABELS[r.policy]}' for i,r in enumerate(main.itertuples())),
            transform=ax.transAxes,va='top',fontsize=9)
    ax.grid(alpha=.2); fig.tight_layout(); save(fig,'risk_growth')
    chosen=main.policy.tolist()
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for axis,metric,title in zip(axes,['median_growth','p90_loss'],['Median growth difference','P90 drawdown difference']):
        g=paired[(paired.horizon==30)&np.isclose(paired.cost_rate,config['cost_rate'])&(paired.metric==metric)].set_index('policy')
        for y,name in enumerate(chosen):
            if name not in g.index: continue
            r=g.loc[name]
            axis.plot([100*r.lower95,100*r.upper95],[y,y],color='#287c8e',lw=2)
            axis.scatter(100*r.difference,y,color='#173e5b',s=24)
        axis.axvline(0,color='gray',ls='--'); axis.set_title(title); axis.set_xlabel('Percentage points vs equal weight')
        axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(chosen)),[LABELS[p] for p in chosen])
    fig.suptitle('Paired scenario bootstrap · pointwise 95% intervals\nExploratory OOF comparisons; fitted-model uncertainty excluded',fontsize=11)
    fig.tight_layout(); save(fig,'paired_intervals')
    selected=[p for p in ['equal_annual','balanced_annual','balanced_hold','cost_aware','adaptive','adaptive_review'] if p in set(main.policy)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name in selected:
        g=summary[(summary.policy==name)&(summary.horizon==30)].sort_values('cost_rate')
        axes[0].plot(100*g.cost_rate,100*g.median_growth,marker='o',label=LABELS[name])
        axes[1].plot(100*g.cost_rate,100*g.mean_trade_rate,marker='o',label=LABELS[name])
    axes[0].set(ylabel='Median real CAGR (%)',xlabel='Cost per transfer (%)',title='Cost sensitivity · frozen parameters')
    axes[1].set(ylabel='Post-initial reviews with trades (%)',xlabel='Cost per transfer (%)',title='Trading efficiency')
    axes[0].legend(fontsize=8); fig.tight_layout(); save(fig,'cost_sensitivity')
    # Choose severe scenario by the official reference only, before comparing strategies on that scenario.
    severe=int(sample[sample.policy=='equal_annual'].sort_values(['real_max_drawdown','scenario_id'],ascending=[False,True]).iloc[0].scenario_id)
    severe_rows=sample[sample.scenario_id==severe].copy(); severe_rows.to_csv(out/'severe_scenario.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4.8))
    for name in selected:
        g=records[(records.policy==name)&(records.scenario_id==severe)&np.isclose(records.cost_rate,config['cost_rate'])].sort_values('period')
        ax.plot([0]+g.period.tolist(),[config['initial_wealth']]+g.real_wealth.tolist(),label=LABELS[name])
        alert=g[g.action=='human_review']
        if len(alert): ax.scatter(alert.period,alert.real_wealth,marker='X',s=90)
    ax.axhline(config['initial_wealth'],color='gray',ls='--',lw=.8)
    ax.set(xlabel='Annual period',ylabel='Real wealth (initial = 100)',title=f'Scenario {severe}: worst reference drawdown in this evaluation')
    ax.legend(fontsize=8); fig.tight_layout(); save(fig,'severe_path')
    fig,ax=plt.subplots(figsize=(10,4.7))
    for name in selected:
        g=summary[(summary.policy==name)&np.isclose(summary.cost_rate,config['cost_rate'])].sort_values('horizon')
        ax.plot(g.horizon,100*g.p90_loss,marker='o',label=LABELS[name])
    ax.set(xlabel='Horizon (years)',ylabel='P90 real maximum drawdown (%)',title='Risk depends on the investment horizon')
    ax.legend(fontsize=8); fig.tight_layout(); save(fig,'horizon_risk')
    columns=['policy','median_growth','p10_growth','median_loss','p90_loss','median_volatility','mean_trade_rate','mean_real_cost','review_rate','pareto']
    display=main[columns].sort_values('median_growth',ascending=False).copy()
    for col in columns[1:]:
        if col not in ('pareto','mean_real_cost'): display[col]=display[col].map(lambda x:f'{100*x:.2f}%')
    display.mean_real_cost=display.mean_real_cost.map(lambda x:f'{x:.3f}')
    # Keep stable machine-facing IDs in raw CSVs; localize presentation only.
    zh_names=dict(zip(LABELS, ['等权年度再平衡','等权买入持有','全持有票据',
        '保守年度再平衡','均衡年度再平衡','成长年度再平衡','均衡买入持有','均衡每三年再平衡',
        '均衡偏离5个百分点再平衡','最小方差','逆波动率','均值方差','CVaR约束配置',
        '固定均衡成本门控','成本感知配置','自适应配置','自适应加人工复核']))
    display['policy']=display['policy'].map(zh_names)
    display['pareto']=display['pareto'].map({True:'是',False:'否'})
    display=display.rename(columns={'policy':'策略','median_growth':'中位实际CAGR','p10_growth':'P10增长',
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
    if (out/'suitability.csv').exists():
        suitability=pd.read_csv(out/'suitability.csv')
        suitability['portfolio']=suitability['portfolio'].replace({'conservative':'保守','balanced':'均衡','growth':'成长','human_review':'人工复核'})
        suitability['liquidity']=suitability['liquidity'].replace({'high':'高','medium':'中','low':'低'})
        suitability['explanation']=suitability['explanation'].replace({
            'Choose the eligible portfolio with highest median real growth; loss tolerance is not guaranteed.':'选择符合筛选条件且中位实际增长最高的组合；不保证损失不超出容忍度。',
            'No portfolio passes the conservative development risk screen and liquidity/horizon rules.':'没有组合同时通过开发风险筛选及流动性/期限规则。'})
        suitability['loss_tolerance']=suitability['loss_tolerance'].map(lambda x:f'{100*x:.0f}%')
        suitability=suitability.rename(columns={'profile_id':'客户','horizon':'期限（年）','loss_tolerance':'损失容忍度',
            'liquidity':'流动性需要','portfolio':'组合/动作','explanation':'说明',
            'final_breach_probability':'最终越界概率','final_median_growth':'最终中位增长'})
        parts+=['<h2>客户入门适配示例</h2><p>对固定组合使用同时的一侧风险界。高流动性只考虑保守组合，'
                '成长组合至少需要15年。容忍度是客户回答，不是保证；20%越界概率筛选是研究假设。</p>',
                suitability.to_html(index=False,escape=True)]
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
