"""Turn completed development decisions into an auditable frozen release choice."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmark')]
from safebench.experiment import verify_bundle,dump
from safebench.policies import make_policy


def main():
    parser=argparse.ArgumentParser(description='核对最后开发决定与冻结客户，生成唯一发布决定')
    parser.add_argument('run',type=Path)
    parser.add_argument('--tests-passed',type=int,required=True)
    args=parser.parse_args();run=args.run.resolve();run.relative_to(ROOT/'benchmark/outputs')
    output=run/'release_decision.json'
    if output.exists(): raise FileExistsError('Release decision exists')
    decision=json.loads((run/'decision.json').read_text())
    if any(all(v.values()) for v in decision['program_gates'].values()):
        raise RuntimeError('A new candidate passed: complete its declared full-development refit and customer screen before release')
    bundle_path=ROOT/'benchmark/submission_frozen_v1.json'
    raw=json.loads(bundle_path.read_text());bundle=verify_bundle(raw)
    lock=ROOT/'benchmark/FINAL_EVALUATION_LOCK.json'
    assert json.loads(lock.read_text())['bundle_sha256']==raw['bundle_sha256']
    final_dir=ROOT/'benchmark/outputs/final_submission_v1'
    final_bundle=json.loads((final_dir/'frozen_bundle.json').read_text())
    assert final_bundle['bundle_sha256']==raw['bundle_sha256']
    saved_final=json.loads((final_dir/'saved_final_verification.json').read_text())
    assert saved_final['all_checks_passed'] and not saved_final['market_data_reopened'] and not saved_final['policies_reevaluated']
    profiles=json.loads((ROOT/'configs/profiles.json').read_text())
    original=list(csv.DictReader((ROOT/'benchmark/outputs/development_submission_v1/selected_suitability.csv').open()))
    customers=[]
    for profile,frozen,selected in zip(profiles,bundle['frozen_suitability'],original):
        assert profile['profile_id']==frozen['profile_id']==selected['profile_id']
        for key in ['horizon','loss_tolerance','liquidity']: assert profile[key]==frozen[key]
        assert frozen['portfolio']==selected['portfolio'] and frozen['policy']==selected['policy']
        row=dict(profile_id=profile['profile_id'],horizon=profile['horizon'],policy=frozen['policy'],portfolio=frozen['portfolio'])
        if frozen['policy']!='human_review':
            p=make_policy(frozen['policy'],bundle['model'],bundle['params'][frozen['policy']],bundle['config'],profile['horizon'])
            row['initial_weights']=list(p.initial_target)
            row['strategic_bill_fraction']=p.initial_target[2]
            assert abs(sum(p.initial_target)-1)<1e-10 and min(p.initial_target)>=0
            if profile['liquidity']=='high': assert p.initial_target[2]>=bundle['config']['submission']['liquidity_floor']-1e-8
        customers.append(row)
    assert len(customers)==5
    customer_result=dict(passed=True,scope='原冻结分配与已核验保存证据；未新评价客户或最终数据',customers=customers,
                         market_data_read=False,policies_simulated=False)
    dump(run/'customer_verification.json',customer_result)
    audited=[]
    for seed in [20261014,20261019]:
        result=json.loads((run/str(seed)/'independent_replay.json').read_text())
        assert result['passed'] and result['short_exact_paths']==960
        audited.append(result)
    manifest=json.loads((run/'manifest.json').read_text())
    protected=manifest['protected_hashes']
    for name,sha in protected.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    dump(output,dict(status='ready',selected_policy=bundle['selected_policy'],
         selection_reason='最后两个候选未同时通过预声明校准、费用与金融采用条件，保留已冻结A。',
         new_research_winner=None,source_research_decision=str((run/'decision.json').relative_to(ROOT)),
         frozen_bundle=str(bundle_path.relative_to(ROOT)),bundle_sha256=raw['bundle_sha256'],
         code_sha256=bundle['code_sha256'],holdout_reused=False,new_final_evaluation=False,
         independent_confirmation_of_research=False,tests_passed=args.tests_passed,
         customer_verification_passed=True,new_research_paths_audited=sum(x['paths'] for x in audited),
         protected_hashes=protected,
         evidence_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [run/'decision.json',run/'customer_verification.json',final_dir/'saved_final_verification.json']}))
    print(output)


if __name__=='__main__': main()
