"""Verify saved selections and apply the predeclared calibration decision gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
from safebench.experiment import dump
from safebench.metrics import paired_comparisons


def independent_selection_check(run, seed, settings, inner):
    """Recompute constraints and ranking without the selection implementation."""
    directory = run/str(seed)
    config = json.loads((directory/'config.json').read_text())
    models = json.loads((directory/'fitted_models.json').read_text())
    splits = json.loads((directory/'splits.json').read_text())
    trials = json.loads((directory/'tuning_trials.json').read_text())
    checks = []
    for item, split in zip(models, splits):
        fold = item['fold']
        assert item['model']['train_ids'] == split['train_ids']
        assert not set(split['train_ids']) & set(split['validation_ids'])
        for audit in split['inner']:
            assert set(audit['train_ids']) | set(audit['validation_ids']) == set(split['train_ids'])
            assert not set(audit['train_ids']) & set(audit['validation_ids'])
            assert audit['model']['train_ids'] == audit['model']['scenario_control']['train_ids'] == audit['train_ids']
        for program, family in settings.get('families',[(settings['diagnostic'],'a'),(settings['primary'],'fusion')]):
            candidates = []
            for row in trials:
                if row['outer_fold'] != fold or row['program'] != program:
                    continue
                if not row['allocation_feasible']:
                    continue
                block = inner[(inner.seed==seed) & (inner.outer_fold==fold) &
                              (inner.family==family) & (inner.trial==row['trial'])]
                assert len(block) == len(split['train_ids'])*len(settings['long_horizons'])
                assert not block.duplicated(['scenario_id','horizon']).any()
                violation = 0.
                for detail in row['horizon_constraints']:
                    b = block[block.horizon==detail['horizon']]
                    assert sorted(b.scenario_id) == sorted(split['train_ids'])
                    losses = sorted((-np.log1p(-b.real_max_drawdown)).tolist(), reverse=True)
                    mass = len(losses)*(1-config['plan_a']['alpha'])
                    whole = int(np.floor(mass))
                    risk = (sum(losses[:whole])+(mass-whole)*losses[whole])/mass
                    budget = b.plan_a_validation_budget.mean()
                    median_loss = b.annual_growth.median()-b.real_cagr.median()
                    p10_loss = b.annual_growth.quantile(.1)-b.real_cagr.quantile(.1)
                    review = b.review.mean()
                    s = config['plan_a']['selection']
                    part = max(0.,(risk-budget-s['log_risk_slack'])/max(budget,.01))
                    part += max(0.,(median_loss-s['median_growth_loss'])/s['median_growth_loss'])
                    part += max(0.,(p10_loss-s['p10_growth_loss'])/s['p10_growth_loss'])
                    part += max(0.,(review-s['maximum_review_rate'])/s['maximum_review_rate'])
                    np.testing.assert_allclose([risk,median_loss,p10_loss,review,part],
                        [detail[k] for k in ['log_risk','median_growth_loss','p10_growth_loss','review_rate','constraint_violation']],atol=1e-10)
                    violation += part
                score = float(np.log1p(block.real_cagr).mean())
                np.testing.assert_allclose([violation,score],[row['constraint_violation'],row['mean_utility']],atol=1e-10)
                candidates.append((row['trial'],violation<=1e-10,violation,score))
            winner = min(candidates,key=lambda x:(not x[1],x[2] if not x[1] else 0.,-x[3],x[0]))
            chosen = item['long_params'][program]
            assert chosen['trial']==winner[0] and chosen['selection_feasible']==winner[1]
            checks.append(dict(seed=seed,outer_fold=fold,program=program,trial=winner[0],
                               selection_feasible=winner[1],constraint_violation=winner[2]))
    return checks, all(d['feasible'] for s in splits for d in s['short_constraints'])


def main():
    parser = argparse.ArgumentParser(description='从保存结果核验长期校准与后续扩展门槛')
    parser.add_argument('run',type=Path)
    run = parser.parse_args().run
    if (run/'decision.json').exists(): raise FileExistsError('Analysis exists')
    manifest = json.loads((run/'manifest.json').read_text())
    settings = manifest['settings']
    inner = pd.read_csv(run/'inner_metrics.csv.gz')
    choices = []
    screens = []
    costs = []
    shorts = []
    references = settings.get('references',['efficient_horizon','plan_a_balanced','balanced_annual','plan_a_annual',
                  'k1_anchor_a','k2_anchor_a','k1_anchor_annual','k2_anchor_annual'])
    for seed in sorted(inner.seed.unique()):
        seed = int(seed)
        directory = run/str(seed)
        checks, short_ok = independent_selection_check(run,seed,settings,inner)
        choices.extend(checks)
        shorts.append(short_ok)
        m = pd.read_csv(directory/'scenario_metrics.csv')
        programs = [settings['diagnostic'],settings['primary']]
        pairs = []
        for ref in references:
            selected = m[m.policy.isin(programs+[ref])]
            p = paired_comparisons(selected,settings['analysis_seed'],settings['bootstrap'],ref)
            p['reference'] = ref
            pairs.append(p)
        pair = pd.concat(pairs,ignore_index=True)
        pair.to_csv(directory/'paired_comparisons.csv',index=False)
        for program in programs:
            for h in settings['long_horizons']:
                for cost in settings['costs']:
                    block = m[(m.horizon==h)&(m.cost_rate==cost)]
                    a = block[block.policy==program]
                    b = block[block.policy==settings.get('cost_baseline','efficient_horizon')]
                    ratio = float(a.total_real_cost.mean()/b.total_real_cost.mean())
                    costs.append(dict(seed=seed,policy=program,horizon=h,cost_rate=cost,cost_ratio=ratio,
                                      passed=ratio<=settings['maximum_cost_ratio']))
            for ref in references:
                for h in settings['horizons']:
                    block = m[(m.horizon==h)&(m.cost_rate==.0025)]
                    a = block[block.policy==program]
                    b = block[block.policy==ref]
                    t = pair[(pair.policy==program)&(pair.reference==ref)&(pair.horizon==h)&(pair.cost_rate==.0025)].set_index('metric')
                    growth,p10,risk = [t.loc[k] for k in ['median_growth','p10_growth','p90_loss']]
                    growth_folds = risk_folds = 0
                    for fold in range(5):
                        af,bf = a[a.fold==fold],b[b.fold==fold]
                        growth_folds += int(af.real_cagr.median()>bf.real_cagr.median())
                        risk_folds += int(af.real_max_drawdown.quantile(.9)<bf.real_max_drawdown.quantile(.9))
                    high = m[(m.horizon==h)&(m.cost_rate==.005)]
                    ah,bh = high[high.policy==program],high[high.policy==ref]
                    high_g = ah.real_cagr.median()-bh.real_cagr.median()
                    high_r = ah.real_max_drawdown.quantile(.9)-bh.real_max_drawdown.quantile(.9)
                    review_ok = a.review.mean()<=.1 and ah.review.mean()<=.1
                    gp = growth.difference>=.001 and growth.family_lower>0 and p10.family_lower>=-.001 and risk.family_upper<=.01 and growth_folds>=4 and high_g>0 and review_ok
                    rp = risk.difference<=-.03 and risk.family_upper<0 and growth.family_lower>=-.005 and p10.family_lower>=-.002 and risk_folds>=4 and high_r<0 and high_g>=-.005 and review_ok
                    screens.append(dict(seed=seed,policy=program,reference=ref,horizon=h,
                        growth_change=growth.difference,growth_lower=growth.family_lower,growth_upper=growth.family_upper,
                        p10_change=p10.difference,p10_lower=p10.family_lower,p10_upper=p10.family_upper,
                        p90_change=risk.difference,p90_lower=risk.family_lower,p90_upper=risk.family_upper,
                        growth_positive_folds=growth_folds,risk_positive_folds=risk_folds,
                        doubled_growth_change=high_g,doubled_p90_change=high_r,
                        growth_pass=bool(gp),risk_pass=bool(rp)))
    pd.DataFrame(choices).to_csv(run/'selection_verification.csv',index=False)
    pd.DataFrame(costs).to_csv(run/'cost_protection.csv',index=False)
    screen = pd.DataFrame(screens)
    screen.to_csv(run/'financial_screen.csv',index=False)
    primary_choices = [c for c in choices if c['program']==settings['primary']]
    primary_costs = [c for c in costs if c['policy']==settings['primary']]
    feasible = all(c['selection_feasible'] for c in primary_choices) and all(shorts)
    cost_ok = all(c['passed'] for c in primary_costs)
    financial = []
    for seed in sorted(inner.seed.unique()):
        table = screen[(screen.seed==seed)&(screen.policy==settings['primary'])&(screen.horizon==30)].set_index('reference')
        parent = table.loc[settings.get('baseline','efficient_horizon')]
        annual = table.loc['balanced_annual']
        anchor = table.loc['k2_anchor_annual']
        financial.append(bool((parent.growth_pass or parent.risk_pass) and
                              (annual.growth_pass or (annual.risk_pass and anchor.risk_pass))))
    decision = dict(selection_independently_verified=True,all_primary_groups_feasible=feasible,
                    primary_cost_protection=cost_ok,short_constraints_passed=all(shorts),
                    primary_financial_pass_both_splits=all(financial),
                    continue_extensions=feasible and cost_ok,
                    replace_frozen_submission=False,independent_confirmation=False,
                    interval_family='每指标/参照/期限/费用/划分内覆盖两个新程序；条件于OOF拟合，不覆盖历史人工搜索。',
                    reason='后续扩展仅在全部校准与费用保护同时通过时执行；当前冻结提交保留。')
    if 'research_priority' in settings:
        gates = {}
        for program in settings['research_priority']:
            selected = [c for c in choices if c['program']==program]
            paid = [c for c in costs if c['policy']==program]
            passed = []
            for seed in sorted(inner.seed.unique()):
                table = screen[(screen.seed==seed)&(screen.policy==program)&(screen.horizon==30)].set_index('reference')
                prefix = 'k2' if program==settings['primary'] else 'k1'
                parent = table.loc[settings['baseline']]
                original = table.loc['efficient_horizon']
                annual = table.loc['balanced_annual']
                anchor = table.loc[prefix+'_anchor_annual']
                other_ok = all(bool(table.loc[ref].growth_pass or table.loc[ref].risk_pass)
                               for ref in settings.get('required_release_references',[]))
                passed.append(bool((parent.growth_pass or parent.risk_pass) and
                    (original.growth_pass or original.risk_pass) and
                    (annual.growth_pass or (annual.risk_pass and anchor.risk_pass)) and other_ok))
            gates[program] = dict(calibrated=all(c['selection_feasible'] for c in selected) and all(shorts),
                                  cost_protected=all(c['passed'] for c in paid),financial_pass=all(passed))
        decision['program_gates'] = gates
        decision['selected_research_policy'] = next((p for p in settings['research_priority'] if all(gates[p].values())), 'efficient_horizon')
        decision['continue_extensions'] = False
        decision['reason'] = '两个预声明独立扩展完成后停止本轮搜索，按固定顺序检查金融/校准/费用条件，不追加组合。'
    dump(run/'decision.json',decision)
    print(json.dumps(decision,ensure_ascii=False,indent=2))
    print(screen[(screen.policy==settings['primary'])&(screen.reference==settings.get('baseline','efficient_horizon'))&(screen.horizon==30)].to_string(index=False))


if __name__ == '__main__': main()
