"""Verify an interface-only repair preserves all development financial metrics."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main(previous,current):
    old,new=Path(previous),Path(current)
    a=json.loads((old/'manifest.json').read_text());b=json.loads((new/'manifest.json').read_text())
    for key in ('phase','scenario_ids','config_sha256','partition_sha256','evaluated_policies'):
        assert a[key]==b[key]
    assert a['phase']=='development' and a['scenario_ids']==list(range(1,121))
    keys=['policy','scenario_id','horizon','cost_rate']
    left=pd.read_csv(old/'scenario_metrics.csv').sort_values(keys).reset_index(drop=True)
    right=pd.read_csv(new/'scenario_metrics.csv').sort_values(keys).reset_index(drop=True)
    columns=[c for c in left if c!='policy_inference_seconds']
    pd.testing.assert_frame_equal(left[columns],right[columns],check_exact=True)
    result=dict(previous=str(old),current=str(new),all_financial_metrics_identical=True,
                metric_rows=len(left),previous_code_sha256=a['code_sha256'],current_code_sha256=b['code_sha256'])
    with (new/'predecessor_equivalence.json').open('x') as file:json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('previous');parser.add_argument('current')
    args=parser.parse_args();main(args.previous,args.current)
