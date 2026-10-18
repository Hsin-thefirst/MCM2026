"""Two predeclared nested calibration comparisons on existing development splits."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
from fusion_calibration import inner_scores, choose_group
from efficient_fusion import EfficientFusion
from fusion_policy import FusionPolicy
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition, folds
from safebench.engine import simulate
from safebench.experiment import dump, code_digest
from safebench.metrics import compute_metrics, summarize
from safebench.policies import fit_model, make_policy
from safebench.plan_a import PlanAPolicy, parameter_grid, allocation_available
from safebench.plan_c import fit_control


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='长期分组及实际融合执行的嵌套开发校准')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT/'benchmark/outputs')
    settings = json.loads((ROOT/'configs/fusion_calibration.json').read_text())
    fusion = json.loads((ROOT/'configs/fusion_research.json').read_text())
    scenarios, data_hash = load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv', 'development')
    lookup = {s.scenario_id: s for s in scenarios}
    assert set(lookup) == set(range(1, 121))
    protected = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'benchmark/FINAL_EVALUATION_LOCK.json',
                 ROOT/'benchmark/submission_frozen_v1.json', ROOT/'dist/SafeInvest_source.zip']}
    sources = [Path(__file__), ROOT/'scripts/fusion_calibration.py', ROOT/'scripts/efficient_fusion.py',
               ROOT/'scripts/fusion_policy.py', ROOT/'scripts/analyze_fusion_calibration.py',
               ROOT/'configs/fusion_calibration.json', ROOT/'configs/fusion_research.json']
    for source, baseline in zip(settings['source_runs'], settings['baseline_runs']):
        sources += [ROOT/source/f for f in ['config.json', 'manifest.json', 'fitted_models.json', 'splits.json', 'tuning_trials.csv']]
        sources.append(ROOT/baseline/'scenario_metrics.csv')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), phase='development',
         holdout_used=False, final_already_exposed=True, independent_confirmation=False, development_sha256=data_hash,
         code_sha256=code_digest(), settings=settings, protected_hashes=protected,
         sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
         selection='K2为预指定主挑战者；K1仅归因。两套已知划分均报告，不按外层选程序。'))
    started = time.perf_counter()
    all_inner = []
    for source_name, baseline_name in zip(settings['source_runs'], settings['baseline_runs']):
        source = ROOT/source_name
        config = json.loads((source/'config.json').read_text())
        for key in ['submission', 'plan_b', 'plan_c']:
            config[key]['enabled'] = False
        seed = config['seed']
        target = out/str(seed)
        target.mkdir()
        dump(target/'config.json', config)
        baseline = pd.read_csv(ROOT/baseline_name/'scenario_metrics.csv')
        fitted = json.loads((source/'fitted_models.json').read_text())
        splits = json.loads((source/'splits.json').read_text())
        source_trials = pd.read_csv(source/'tuning_trials.csv')
        source_manifest = json.loads((source/'manifest.json').read_text())
        assert source_manifest['code_sha256'] == code_digest()
        assert source_manifest.get('partition_sha256', source_manifest.get('development_sha256')) == data_hash
        metrics = []
        trials = []
        models = []
        audit = []
        annual_records = cap_checks = gate_checks = short_equal = original_equal = 0
        with gzip.open(target/'decisions.jsonl.gz', 'wt', encoding='utf-8') as ledger:
            for item, split, valid_ids in zip(fitted, splits, folds(lookup, 5, seed)):
                fold = item['fold']
                assert valid_ids == split['validation_ids']
                model = item['model']
                ids = split['train_ids']
                assert model['train_ids'] == ids and not set(ids) & set(valid_ids)
                model['scenario_control'] = fit_control([lookup[sid] for sid in ids], config)
                original = item['params']['plan_a_balanced']
                grid = parameter_grid(config)
                trial_id = next(i for i,p in enumerate(grid) if all(original[k] == p[k] for k in p))
                saved = source_trials[(source_trials.policy == 'plan_a_balanced') &
                                     (source_trials.outer_fold == fold) & (source_trials.trial == trial_id)].iloc[0]
                short_constraints = [r for r in json.loads(saved.horizon_constraints) if r['horizon'] <= 6]
                assert {r['horizon'] for r in short_constraints} == {5,6}
                inner_rows = []
                unavailable = {}
                inner_audit = []
                for inner, inner_valid in enumerate(folds(ids, 3, seed+fold+1)):
                    train_ids = [sid for sid in ids if sid not in inner_valid]
                    saved_inner = split['inner'][inner]
                    assert train_ids == saved_inner['train_ids'] and inner_valid == saved_inner['validation_ids']
                    inner_model = fit_model([lookup[sid] for sid in train_ids], config)
                    inner_model['scenario_control'] = fit_control([lookup[sid] for sid in train_ids], config)
                    assert inner_model['train_ids'] == train_ids
                    inner_audit.append(dict(inner_fold=inner, train_ids=train_ids, validation_ids=inner_valid, model=inner_model))
                    for trial, params in enumerate(grid):
                        if not allocation_available(inner_model, 'balanced', params, settings['long_horizons']):
                            unavailable.setdefault(trial, []).append(inner)
                            continue
                        rows = inner_scores([lookup[sid] for sid in inner_valid], inner_model, params, config,
                                            settings['long_horizons'], fusion, settings['rule'])
                        inner_rows.extend(dict(r, trial=trial, inner_fold=inner) for r in rows)
                    print(f'seed={seed} 外折{fold+1}/5 内折{inner+1}/3完成', flush=True)
                frame = pd.DataFrame(inner_rows)
                selected = {}
                for family, name in [('a', settings['diagnostic']), ('fusion', settings['primary'])]:
                    params, evidence = choose_group(frame, model, config, settings['long_horizons'], ids, family, unavailable)
                    selected[name] = params
                    trials.extend(dict(r, outer_fold=fold, program=name) for r in evidence)
                all_inner.extend(dict(r, seed=seed, outer_fold=fold) for r in inner_rows)
                models.append(dict(fold=fold, model=model, original=original, long_params=selected))
                audit.append(dict(outer_fold=fold, train_ids=ids, validation_ids=valid_ids,
                                  short_constraints=short_constraints, inner=inner_audit))
                dump(target/'fitted_models.json', models)
                dump(target/'splits.json', audit)
                print(f'seed={seed} 外折{fold+1} 长期选择：{selected}', flush=True)
                names = ['efficient_horizon', settings['diagnostic'], settings['primary'], 'plan_a_balanced',
                         'fusion_drawdown', 'equal_annual', 'balanced_annual', 'plan_a_annual', 'plan_a_hold']
                names += [f'{prefix}_anchor_{mode}' for prefix in ['k1','k2'] for mode in ['a','annual','hold']]
                for name in names:
                    for cost in settings['costs']:
                        for h in settings['horizons']:
                            chosen = original
                            if h > 6:
                                if name == settings['diagnostic'] or name.startswith('k1_'):
                                    chosen = selected[settings['diagnostic']]
                                elif name == settings['primary'] or name.startswith('k2_'):
                                    chosen = selected[settings['primary']]
                            for sid in valid_ids:
                                if name in ['efficient_horizon', settings['diagnostic'], settings['primary']]:
                                    policy = EfficientFusion(name, model, chosen, config, h, fusion, settings['rule'])
                                elif '_anchor_' in name:
                                    mode = name.rsplit('_',1)[1]
                                    policy = PlanAPolicy(name, model, chosen, config, h,
                                                        mode='partial' if mode == 'a' else mode, review=mode == 'a')
                                elif name == 'fusion_drawdown':
                                    policy = FusionPolicy(name, model, original, config, h, fusion)
                                else:
                                    policy = make_policy(name, model, chosen, config, h)
                                result = simulate(lookup[sid], policy, policy.initial_target,
                                                  SimulationConfig(h, config['initial_wealth'], cost))
                                row = dict(compute_metrics(result), fold=fold)
                                metrics.append(row)
                                if name == 'efficient_horizon':
                                    old = baseline[(baseline.policy == name) & (baseline.scenario_id == sid) &
                                                   (baseline.horizon == h) & (baseline.cost_rate == cost)].iloc[0]
                                    for key in compute_metrics(result):
                                        if key not in ['policy','scenario_id','horizon','cost_rate']:
                                            np.testing.assert_allclose(row[key], old[key], atol=1e-12, rtol=1e-12)
                                    original_equal += 1
                                if h <= 6 and name in [settings['diagnostic'], settings['primary']]:
                                    old = next(r for r in reversed(metrics) if r['policy']=='efficient_horizon' and
                                               r['scenario_id']==sid and r['horizon']==h and r['cost_rate']==cost)
                                    for key in compute_metrics(result):
                                        if key != 'policy': assert row[key] == old[key]
                                    short_equal += 1
                                for record in result.records:
                                    if record.period > 1 and record.moved_amount > 1e-8 and policy.requires_benefit_gate:
                                        assert record.estimated_benefit > record.estimated_cost+policy.buffer
                                        gate_checks += 1
                                    if record.policy_diagnostics and name in ['efficient_horizon',settings['diagnostic'],settings['primary']]:
                                        before = np.asarray(record.pre_trade_holdings)
                                        after = np.asarray(record.post_trade_holdings)
                                        assert .5*np.abs(before/before.sum()-after/after.sum()).sum() <= .1+1e-10
                                        cap_checks += 1
                                    r = asdict(record)
                                    r.update(action=record.action.value, policy=name, scenario_id=sid, horizon=h,
                                             cost_rate=cost, fold=fold, required_buffer=getattr(policy,'buffer',0.),
                                             requires_benefit_gate=policy.requires_benefit_gate)
                                    ledger.write(json.dumps(r, allow_nan=False)+'\n')
                                    annual_records += 1
                print(f'seed={seed} 外折{fold+1}/5 外层评价完成', flush=True)
        table = pd.DataFrame(metrics)
        table.to_csv(target/'scenario_metrics.csv', index=False)
        summarize(table).to_csv(target/'summary.csv', index=False)
        dump(target/'tuning_trials.json', trials)
        manifest = json.loads((out/'manifest.json').read_text())
        dump(target/'manifest.json', dict(manifest, seed=seed, source_run=source_name))
        dump(target/'verification.json', dict(passed=True, paths=len(table), annual_records=annual_records,
             original_matched_paths=original_equal, short_matched_paths=short_equal,
             gate_checks=gate_checks, cap_checks=cap_checks))
    pd.DataFrame(all_inner).to_csv(out/'inner_metrics.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    for path, digest in protected.items():
        assert sha(ROOT/path) == digest
    dump(out/'verification.json', dict(passed=True, protected_unchanged=True,
         inner_paths=len(all_inner), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
