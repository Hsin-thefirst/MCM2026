"""Review immutable B/C development outputs and prepare report datasets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'benchmark/outputs/development_plan_b_v3'
C = ROOT/'benchmark/outputs/development_plan_c_screen_v3'
LABELS = {
 'plan_a_balanced':'A 均衡基线', 'plan_b_narrow':'B 窄动作·内层所选',
 'plan_b_wide':'B 宽动作·内层所选', 'plan_b_wide_05':'B 宽动作·固定半强度',
 'plan_c_l1':'C 一步·含风险', 'plan_c_l3':'C 三步·含风险',
 'plan_c_l1_no_risk':'C 一步·无风险项', 'plan_c_l3_no_risk':'C 三步·无风险项',
 'balanced_annual':'原均衡·年度', 'balanced_hold':'原均衡·持有',
 'equal_annual':'等权·年度', 'plan_a_annual':'A 同锚·年度',
 'plan_b_narrow_0':'B 窄动作·零信号', 'plan_b_narrow_025':'B 窄动作·四分之一强度',
 'plan_b_narrow_05':'B 窄动作·半强度', 'plan_b_wide_0':'B 宽动作·零信号',
 'plan_b_wide_025':'B 宽动作·四分之一强度', 'plan_b_period':'B 仅时间趋势',
 'plan_b_shifted':'B 错位标签',
}

def rows(frame):
    return json.loads(frame.to_json(orient='records',force_ascii=False,double_precision=15))

def label(frame):
    return frame.assign(strategy=frame.policy.map(lambda p:LABELS.get(p,p)))

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    frames={};summaries={};sources={}
    for name,path in [('B',B),('C',C)]:
        manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['phase']=='development' and manifest['scenario_ids']==list(range(1,121))
        assert manifest['holdout_used'] is False
        for proof in ('validation.json','extension_verification.json','independent_replay.json'):
            assert json.loads((path/proof).read_text())['all_checks_passed']
        m=pd.read_csv(path/'scenario_metrics.csv');s=pd.read_csv(path/'summary.csv')
        assert not m.duplicated(['policy','scenario_id','horizon','cost_rate']).any()
        assert set(m.scenario_id)==set(range(1,121))
        for (p,h,cost),g in m.groupby(['policy','horizon','cost_rate']):
            r=s[(s.policy==p)&(s.horizon==h)&np.isclose(s.cost_rate,cost)].iloc[0]
            np.testing.assert_allclose(
                [g.real_cagr.median(),g.real_cagr.quantile(.1),g.real_max_drawdown.quantile(.9),g.total_real_cost.mean(),g.trade_rate.mean()],
                [r.median_growth,r.p10_growth,r.p90_loss,r.mean_real_cost,r.mean_trade_rate],atol=1e-14)
        frames[name]=m;summaries[name]=s
        for filename in ('manifest.json','config.json','scenario_metrics.csv','summary.csv','extension_screen.csv','extension_comparisons.csv','fitted_models.json','allocation_exposure.csv','validation.json','independent_replay.json','extension_verification.json'):
            file=path/filename;sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    a,b=frames['B'],frames['C'];keys=['policy','scenario_id','horizon','cost_rate']
    common=a.merge(b,on=keys,suffixes=('_b','_c'),validate='one_to_one')
    for col in ('real_cagr','real_max_drawdown','total_real_cost','trade_rate'):
        np.testing.assert_array_equal(common[col+'_b'],common[col+'_c'])
    metrics=pd.concat([a,b[b.policy.str.startswith('plan_c_')]],ignore_index=True)
    summary=label(pd.concat([summaries['B'],summaries['C'][summaries['C'].policy.str.startswith('plan_c_')]],ignore_index=True))
    main=summary[(summary.horizon==30)&np.isclose(summary.cost_rate,.0025)]
    core=list(LABELS)[:12];selected=core[:6]
    comparison=pd.concat([pd.read_csv(B/'extension_screen.csv').assign(run='B'),pd.read_csv(C/'extension_screen.csv').assign(run='C')])
    comparison=label(comparison);comparison['reference_name']=comparison.reference.map(LABELS)
    mm=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,.0025)]
    folds=[]
    for (p,f),g in mm[mm.policy.isin(core)].groupby(['policy','fold']):
        folds.append(dict(policy=p,strategy=LABELS[p],fold=int(f)+1,n=len(g),median_growth=g.real_cagr.median(),p90_loss=g.real_max_drawdown.quantile(.9),mean_log_growth=np.log1p(g.real_cagr).mean()))
    paired=[];tails=[]
    for candidate,reference in [('plan_b_narrow','plan_b_narrow_0'),('plan_b_wide','plan_b_wide_0'),('plan_c_l3','plan_c_l1'),('plan_c_l1','plan_a_balanced'),('plan_c_l3','plan_a_balanced')]:
        l=mm[mm.policy==candidate].set_index('scenario_id').sort_index()
        r=mm[mm.policy==reference].set_index('scenario_id').sort_index()
        assert l.index.equals(r.index)
        d=l.real_cagr-r.real_cagr;dd=l.real_max_drawdown-r.real_max_drawdown
        paired.append(dict(policy=candidate,reference=reference,strategy=LABELS[candidate],reference_name=LABELS[reference],n=len(d),growth_wins=int((d>1e-14).sum()),growth_ties=int((d.abs()<=1e-14).sum()),growth_losses=int((d<-1e-14).sum()),median_paired_growth_difference=d.median(),mean_paired_growth_difference=d.mean(),difference_of_growth_medians=l.real_cagr.median()-r.real_cagr.median(),mean_log_growth_difference=(np.log1p(l.real_cagr)-np.log1p(r.real_cagr)).mean(),mean_drawdown_difference=dd.mean()))
        if candidate=='plan_c_l3' and reference=='plan_c_l1':
            for sid in dd.nlargest(5).index:
                tails.append(dict(scenario_id=int(sid),c1_growth=r.loc[sid,'real_cagr'],c3_growth=l.loc[sid,'real_cagr'],c1_drawdown=r.loc[sid,'real_max_drawdown'],c3_drawdown=l.loc[sid,'real_max_drawdown'],drawdown_difference=dd.loc[sid]))
    ec=pd.read_csv(C/'allocation_exposure.csv')
    exposures=pd.concat([pd.read_csv(B/'allocation_exposure.csv'),ec[ec.policy.str.startswith('plan_c_')]])
    exposure=label(exposures[(exposures.simulation_horizon==30)&np.isclose(exposures.cost_rate,.0025)&exposures.policy.isin(core)])
    inner=pd.read_csv(B/'signal_inner_constraints.csv');chosen=[]
    for f in json.loads((B/'fitted_models.json').read_text()):
        for family in ('narrow','wide'):
            p=f['params']['plan_b_'+family]
            chosen.append(dict(fold=f['fold']+1,family=family,eta=p['eta'],feasible=p['signal_selection_feasible']))
    trades=[];actions=[]
    for path,policies in [(B,selected[:4]),(C,core[4:8])]:
        columns=['policy','scenario_id','simulation_horizon','period','cost_rate','moved_amount','transaction_cost','pre_trade_holdings_equity','pre_trade_holdings_bond','pre_trade_holdings_bill','policy_diagnostics']
        chunks=[]
        for g in pd.read_csv(path/'decisions.csv.gz',usecols=columns,chunksize=120000):
            chunks.append(g[g.policy.isin(policies)&(g.simulation_horizon==30)&np.isclose(g.cost_rate,.0025)&(g.period>1)])
        records=pd.concat(chunks,ignore_index=True)
        records['fraction']=records.moved_amount/records[[c for c in columns if c.startswith('pre_trade_holdings_')]].sum(axis=1)
        for p,g in records.groupby('policy'):
            t=g[g.fraction>1e-10]
            trades.append(dict(policy=p,strategy=LABELS[p],reviews=len(g),trades=len(t),trade_rate=len(t)/len(g),mean_fraction_when_trade=t.fraction.mean(),median_fraction_when_trade=t.fraction.median(),p90_fraction_when_trade=t.fraction.quantile(.9),mean_postinitial_turnover=g.fraction.mean()))
            if p.startswith('plan_c_'):
                for r in t.itertuples():
                    actions.append(dict(policy=p,first_type=json.loads(r.policy_diagnostics)['chosen_sequence'][0],moved_fraction=r.fraction,nominal_fee=r.transaction_cost))
    action=pd.DataFrame(actions).groupby(['policy','first_type']).agg(trades=('first_type','size'),mean_moved_fraction=('moved_fraction','mean'),nominal_fee=('nominal_fee','sum')).reset_index()
    tables=dict(overview=main[main.policy.isin(core)],all_strategies=main,comparisons=comparison,folds=pd.DataFrame(folds),paired=pd.DataFrame(paired),tail_cases=pd.DataFrame(tails),exposures=exposure,signals=pd.read_csv(B/'signal_diagnostics.csv'),inner_checks=inner.groupby(['family','eta','horizon']).agg(checks=('feasible','size'),passes=('feasible','sum')).reset_index(),inner_failures=inner[~inner.feasible],selected_eta=pd.DataFrame(chosen),trade_sizes=pd.DataFrame(trades),control_actions=action,horizons=summary[summary.policy.isin(selected)&np.isclose(summary.cost_rate,.0025)],costs=summary[(summary.horizon==30)&summary.policy.isin(selected)],signal_candidates=main[main.policy.str.startswith('plan_b_')])
    queries={}
    for name,frame in tables.items():
        frame.to_csv(out/(name+'.csv'),index=False)
        queries[name]=dict(rows=rows(frame),source=dict(type='file',title='SafeInvest 已核验开发输出',files=[str((B/'scenario_metrics.csv').relative_to(ROOT)),str((C/'scenario_metrics.csv').relative_to(ROOT))],period='合成路径年1–30，不是公历年度',scope='情景1–120；5外折×3内折；C仅H30正常/翻倍费用。',caveats=['折外开发估计，真实121–180未使用。','共享策略去重，多个策略行不等于多个独立样本。','区间复用已登记多候选调整bootstrap；条件于拟合模型，未覆盖全部重拟合与重复研究不确定性。'],metricDefinitions=[dict(label='增长',definition='逐情景终值实际财富/100的1/H次方减1，再跨120条情景取中位或P10。'),dict(label='P90回撤',definition='各路径含费用时点的最大实际回撤的90%分位数。'),dict(label='实际费用',definition='各次名义费用除以当时累计通胀后求和；包含建仓，初始财富100，不是年费率或终值机会损失。'),dict(label='差值',definition='候选减指定参照，小数乘100为百分点；中位数之差不同于逐情景差值的中位数。')],evidenceFlow=[dict(title='输入与核验',detail='读取B/C v3已存输出和核验JSON，逐组独立复算汇总，并检验共有策略一致。'),dict(title='派生分析',detail='scripts/analyze_bc_results.py；仅按已存情景/策略/期限/费用重组数据，计算折分布、交易幅度与暴露；不拟合或重选模型。')]),methods=[dict(language='python',code='python scripts/analyze_bc_results.py --out <new_directory>')])
    snapshot=dict(id='mcm-safeinvest-bc-results-20261017',surface='report',title='B的弱预测尚未转化为稳定收益，C加深规划没有改善主比较',buildStatus='creating',generatedAt=datetime.now(timezone.utc).isoformat(),report=dict(asOf='2026-10-17'),queries=queries)
    (out/'reviewed.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    (out/'evidence_manifest.json').write_text(json.dumps(dict(phase='development_saved_outputs_only',holdout_used=False,source_sha256=sources,shared_rows_verified=len(common),B_metric_rows=len(a),C_metric_rows=len(b),unique_merged_rows=len(metrics),aggregates_verified=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),shared_rows_verified=len(common),unique_merged_rows=len(metrics),tables={k:len(v) for k,v in tables.items()}),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
