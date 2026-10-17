"""Fixed, post-final exploratory OOF fusion experiment; never a new holdout test."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
from fusion_policy import FusionPolicy
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import dump, code_digest
from safebench.metrics import compute_metrics, summarize, paired_comparisons
from safebench.plan_a import PlanAPolicy


def scalar_check(result, scenario):
    """Independently reconstruct each declared action and nominal/real accounting."""
    holdings = [0., 0., result.config.initial_wealth]
    inflation = 1.; peak = result.config.initial_wealth
    maximum = fees_real = 0.; error = 0.; gates = caps = 0
    for r, obs in zip(result.records, scenario.observations):
        old = sum(holdings)
        trading = r.action.value in ('rebalance', 'change_target') or r.period == 1
        target = list(r.target_weights) if trading else [x/old for x in holdings]
        moved = sum(abs(old*w-v) for w, v in zip(target, holdings))/2
        fee = moved*result.config.transaction_cost_rate
        post = [(old-fee)*w for w in target] if trading else holdings[:]
        np.testing.assert_allclose(r.post_trade_holdings, post, atol=1e-9, rtol=1e-12)
        fees_real += fee/inflation
        maximum = max(maximum, 1-sum(post)/inflation/peak)
        peak = max(peak, sum(post)/inflation)
        holdings = [v*(1+ret) for v, ret in zip(post, obs.nominal_returns)]
        inflation *= 1+obs.inflation
        real = sum(holdings)/inflation
        maximum = max(maximum, 1-real/peak); peak = max(peak, real)
        error = max(error, abs(real-r.real_wealth), abs(fee-r.transaction_cost))
        np.testing.assert_allclose(r.end_holdings, holdings, atol=1e-9, rtol=1e-12)
        if r.period > 1 and trading:
            if not r.estimated_benefit > r.estimated_cost + .0001:
                raise AssertionError('Benefit gate failed')
            gates += 1
            if result.policy_name.startswith('fusion_'):
                if moved/old > .1+1e-10: raise AssertionError('Move cap failed')
                caps += 1
    m = compute_metrics(result)
    np.testing.assert_allclose([m['real_cagr'], m['real_max_drawdown'], m['total_real_cost']],
        [(real/result.config.initial_wealth)**(1/result.config.horizon)-1, maximum, fees_real], atol=1e-10)
    return error, gates, caps


def main():
    parser = argparse.ArgumentParser(description='固定三策略融合开发研究，禁止最终评价')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.relative_to(ROOT/'benchmark/outputs')
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads((ROOT/'configs/fusion_research.json').read_text())
    source = ROOT/settings['source_run']
    config = json.loads((source/'config.json').read_text())
    models = json.loads((source/'fitted_models.json').read_text())
    splits = json.loads((source/'splits.json').read_text())
    scenarios, sha = load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv', 'development')
    source_manifest = json.loads((source/'manifest.json').read_text())
    assert source_manifest['partition_sha256'] == sha
    assert source_manifest['code_sha256'] == code_digest()
    lookup = {s.scenario_id: s for s in scenarios}
    assert set(lookup) == set(range(1, 121))
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    protected = [ROOT/'benchmark/FINAL_EVALUATION_LOCK.json', ROOT/'benchmark/submission_frozen_v1.json']
    protected_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase='development',
        holdout_used=False, final_already_exposed=True, independent_confirmation=False,
        development_sha256=sha, settings=settings, source_core_sha256=code_digest(),
        sources={str(p.relative_to(ROOT)):digest(p) for p in [source/'fitted_models.json',source/'splits.json',
            source/'scenario_metrics.csv', ROOT/'configs/fusion_research.json',Path(__file__),ROOT/'scripts/fusion_policy.py']},
        protected_hashes=protected_hashes, selection='三条预固定规则全部报告；不按折外结果替换提交版。')
    dump(out/'manifest.json', manifest)
    started = time.perf_counter(); rows = []; ledger = []; errors = []; gates = caps = 0
    for fitted, split in zip(models, splits):
        model = fitted['model']; params = fitted['params']['plan_a_balanced']
        assert fitted['fold'] == split['outer_fold']
        assert model['train_ids'] == split['train_ids']
        assert not set(model['train_ids']) & set(split['validation_ids'])
        assert set(model['train_ids']) | set(split['validation_ids']) == set(lookup)
        for name in ['plan_a_balanced']+settings['policies']:
            for cost in settings['costs']:
                for h in settings['horizons']:
                    for sid in split['validation_ids']:
                        p = (PlanAPolicy(name, model, params, config, h) if name == 'plan_a_balanced' else
                             FusionPolicy(name, model, params, config, h, settings))
                        result = simulate(lookup[sid], p, p.initial_target, SimulationConfig(h, 100., cost))
                        err, ng, nc = scalar_check(result, lookup[sid]); errors.append(err); gates += ng; caps += nc
                        rows.append(dict(compute_metrics(result), fold=fitted['fold']))
                        for r in result.records:
                            row = asdict(r)
                            for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                                values = row.pop(field)
                                row.update({field+'_'+a:v for a,v in zip(('equity','bond','bill'),values)})
                            row.update(policy=name, scenario_id=sid, cost_rate=cost, simulation_horizon=h, fold=fitted['fold'])
                            row['action'] = r.action.value
                            row['reason_codes'] = json.dumps(row['reason_codes'])
                            row['policy_diagnostics'] = json.dumps(row['policy_diagnostics'])
                            ledger.append(row)
            print(f'外折 {fitted["fold"]+1}/5：{name} 完成', flush=True)
    frame = pd.DataFrame(rows)
    old = pd.read_csv(source/'scenario_metrics.csv')
    keys = ['policy', 'scenario_id', 'horizon', 'cost_rate']
    a = frame[frame.policy == 'plan_a_balanced'].set_index(keys).sort_index()
    b = old.set_index(keys).loc[a.index].sort_index()
    common = [c for c in a.columns if c != 'fold']
    pd.testing.assert_frame_equal(a[common], b[common], check_exact=False, atol=1e-12, rtol=1e-12, check_dtype=False)
    refs = old[old.policy.isin(['equal_annual','balanced_annual','plan_a_annual','submission_c_cap10']) & old.cost_rate.isin(settings['costs'])]
    combined = pd.concat([frame, refs[frame.columns]], ignore_index=True)
    combined.to_csv(out/'scenario_metrics.csv', index=False)
    summarize(combined).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(ledger).to_csv(out/'decisions.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    comparisons = []
    for ref in ['plan_a_balanced','fusion_mpc','equal_annual','balanced_annual']:
        table = paired_comparisons(combined,settings['seed'],settings['bootstrap'],ref)
        table['reference'] = ref; comparisons.append(table)
    pd.concat(comparisons,ignore_index=True).to_csv(out/'paired_comparisons.csv',index=False)
    for p in protected: assert digest(p) == protected_hashes[str(p.relative_to(ROOT))]
    dump(out/'verification.json',dict(passed=True, evaluated_paths=len(frame), annual_records=len(ledger),
        original_a_matched_rows=len(a), benefit_gate_checks=gates, cap_checks=caps,
        maximum_scalar_error=max(errors), final_lock_and_bundle_unchanged=True,
        seconds=time.perf_counter()-started))
    print(summarize(combined).query('horizon == 30 and cost_rate == 0.0025').to_string(index=False), flush=True)


if __name__ == '__main__': main()
