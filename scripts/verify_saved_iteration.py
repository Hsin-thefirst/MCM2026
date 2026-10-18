"""Independent scalar reconstruction of all saved development iteration paths."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmark')]
from safebench.data import load_partition
from safebench.experiment import dump


def main():
    parser=argparse.ArgumentParser(description='独立重放已保存迭代账本，只读取开发市场')
    parser.add_argument('run',type=Path);args=parser.parse_args();run=args.run
    target=run/'independent_replay.json'
    if target.exists():raise FileExistsError('Audit exists')
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['phase']=='development' and not manifest['holdout_used']
    scenarios,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv','development')
    assert sha==manifest['development_sha256']
    lookup={s.scenario_id:s for s in scenarios}
    metrics=pd.read_csv(run/'scenario_metrics.csv').set_index(['policy','scenario_id','horizon','cost_rate'])
    seen=set();active=None;count=paths=0;maximum_error=0.
    def check(values,expected):
        nonlocal maximum_error
        maximum_error=max(maximum_error,float(np.max(np.abs(np.asarray(values)-expected))))
        np.testing.assert_allclose(values,expected,atol=2e-9,rtol=2e-12)
    with gzip.open(run/'decisions.jsonl.gz','rt') as handle:
        for line in handle:
            r=json.loads(line); key=(r['policy'],r['scenario_id'],r['horizon'],r['cost_rate']);t=r['period']
            if key!=active:
                if active is not None:assert previous_period==active[2]
                assert key not in seen and t==1;seen.add(key);active=key
                holdings=[0.,0.,100.];inflation=1.;peak=100.;mdd=fee_sum=0.;trades=0;previous_period=0
            assert t==previous_period+1;previous_period=t
            check(r['pre_trade_holdings'],holdings)
            wealth=sum(holdings)
            if t==1 or r['action'] in ('rebalance','change_target'):
                weights=r['target_weights'];moved=sum(abs(wealth*w-v) for w,v in zip(weights,holdings))/2
                fee=moved*r['cost_rate'];holdings=[(wealth-fee)*w for w in weights]
            else:moved=fee=0.
            check([r['moved_amount'],r['transaction_cost']],[moved,fee])
            check(r['post_trade_holdings'],holdings)
            fee_sum+=fee/inflation
            mark=sum(holdings)/inflation;mdd=max(mdd,1-mark/peak);peak=max(peak,mark)
            obs=lookup[r['scenario_id']].observations[t-1]
            holdings=[v*(1+g) for v,g in zip(holdings,obs.nominal_returns)]
            inflation*=1+obs.inflation
            mark=sum(holdings)/inflation;mdd=max(mdd,1-mark/peak);peak=max(peak,mark)
            check(r['end_holdings'],holdings);check([r['real_wealth'],r['cumulative_inflation']],[mark,inflation])
            if t>1 and moved>1e-8:trades+=1
            if t==r['horizon']:
                expected=metrics.loc[key]
                check([expected.real_cagr,expected.real_max_drawdown,expected.total_real_cost,expected.trade_rate],
                    [(mark/100)**(1/t)-1,mdd,fee_sum,trades/(t-1)])
                paths+=1
            count+=1
    assert previous_period==active[2]
    declared=json.loads((run/'verification.json').read_text())
    assert count==declared['annual_records'] and paths==declared['paths']
    dump(target,dict(passed=True,paths=paths,annual_records=count,maximum_error=maximum_error,
        partition='development_1_120',new_policy_evaluation=False,final_used=False))
    print(target,paths,count,maximum_error)


if __name__=='__main__':main()
