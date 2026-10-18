"""Scalar market replay and exact short-path reproduction for saved calibration."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
from safebench.data import load_partition
from safebench.experiment import dump


def main():
    parser = argparse.ArgumentParser(description='标量重放校准账本并逐年度核验短期路径')
    parser.add_argument('run',type=Path)
    run = parser.parse_args().run
    output = run/'independent_replay.json'
    if output.exists(): raise FileExistsError('Audit exists')
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['phase']=='development' and not manifest['holdout_used']
    scenarios,sha = load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv','development')
    assert sha==manifest['development_sha256']
    lookup = {s.scenario_id:s for s in scenarios}
    with (run/'scenario_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    metrics = {(r['policy'],int(r['scenario_id']),int(r['horizon']),float(r['cost_rate'])):r for r in rows}
    assert len(metrics)==len(rows)==18000
    names = {r['policy'] for r in rows}
    assert len(names)==15
    expected = {(name,sid,h,c) for name in names for sid in range(1,121)
                for h in manifest['settings']['horizons'] for c in manifest['settings']['costs']}
    assert set(metrics)==expected
    signatures = {}
    active = None
    seen = set()
    count = gates = caps = short_equal = 0
    error = 0.
    fingerprint_fields = ['period','action','target_weights','pre_trade_holdings','post_trade_holdings',
        'end_holdings','moved_amount','transaction_cost','cumulative_inflation','real_wealth','review_pending',
        'reason_codes','explanation','estimated_benefit','estimated_cost']
    def check(a,b):
        nonlocal error
        if isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b): check(x,y)
        else:
            error=max(error,abs(float(a)-float(b)))
            assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-9),(a,b)
    with gzip.open(run/'decisions.jsonl.gz','rt') as handle:
        for line in handle:
            r=json.loads(line)
            key=(r['policy'],r['scenario_id'],r['horizon'],r['cost_rate'])
            t=r['period']
            if key != active:
                if active is not None: assert previous==active[2]
                assert key in expected and key not in seen and t==1
                active=key
                seen.add(key)
                holdings=[0.,0.,100.]
                inflation=1.
                peak=100.
                mdd=real_fees=0.
                previous=trades=0
                pending=False
                fingerprint=hashlib.sha256()
            assert t==previous+1
            previous=t
            if pending: assert r['review_pending'] and r['action']=='keep'
            pending=r['review_pending']
            check(r['pre_trade_holdings'],holdings)
            wealth=sum(holdings)
            if t==1 or r['action'] in ['rebalance','change_target']:
                w=r['target_weights']
                check(sum(w),1.)
                assert min(w)>=0
                moved=sum(abs(wealth*x-v) for x,v in zip(w,holdings))/2
                fee=moved*key[3]
                post=[(wealth-fee)*x for x in w]
            else:
                moved=fee=0.
                post=holdings[:]
            if t>1 and moved>1e-8:
                trades+=1
                if r['requires_benefit_gate']:
                    assert r['estimated_benefit']>r['estimated_cost']+r['required_buffer']
                    check(r['estimated_cost'],fee/wealth)
                    gates+=1
                if key[0] in ['efficient_horizon','calibration_group_a','calibration_group_fusion','fusion_drawdown'] and key[2]>6:
                    assert moved/wealth<=.1+1e-10
                    caps+=1
            check([r['moved_amount'],r['transaction_cost']],[moved,fee])
            check(r['post_trade_holdings'],post)
            real_fees+=fee/inflation
            mark=sum(post)/inflation
            mdd=max(mdd,1-mark/peak)
            peak=max(peak,mark)
            obs=lookup[key[1]].observations[t-1]
            holdings=[v*(1+g) for v,g in zip(post,obs.nominal_returns)]
            inflation*=1+obs.inflation
            mark=sum(holdings)/inflation
            mdd=max(mdd,1-mark/peak)
            peak=max(peak,mark)
            check(r['end_holdings'],holdings)
            check([r['real_wealth'],r['cumulative_inflation']],[mark,inflation])
            if key[0] in ['efficient_horizon','calibration_group_a','calibration_group_fusion'] and key[2]<=6:
                fingerprint.update(json.dumps({k:r[k] for k in fingerprint_fields},sort_keys=True).encode())
            if t==key[2]:
                metric=metrics[key]
                check([float(metric[k]) for k in ['real_cagr','real_max_drawdown','total_real_cost','trade_rate']],
                      [(mark/100)**(1/t)-1,mdd,real_fees,trades/(t-1)])
                if key[0] in ['efficient_horizon','calibration_group_a','calibration_group_fusion'] and key[2]<=6:
                    short_key=key[1:]
                    if key[0]=='efficient_horizon': signatures[short_key]=fingerprint.hexdigest()
                    else:
                        assert signatures[short_key]==fingerprint.hexdigest()
                        short_equal+=1
            count+=1
    assert previous==active[2] and seen==expected
    declared=json.loads((run/'verification.json').read_text())
    assert count==declared['annual_records'] and len(seen)==declared['paths'] and short_equal==960
    dump(output,dict(passed=True,paths=len(seen),annual_records=count,maximum_error=error,
        short_exact_paths=short_equal,gate_checks=gates,cap_checks=caps,final_used=False,
        new_policy_evaluation=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(json.loads(output.read_text()),indent=2))


if __name__=='__main__': main()
