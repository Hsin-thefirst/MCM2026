"""Independent market replay and closed-form reference check; development data only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark'))
import json
import numpy as np
import pandas as pd
from safebench import ROOT
from safebench.data import load_partition

out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'benchmark/outputs/development_consolidated'
if json.loads((out/'manifest.json').read_text())['phase']!='development':
    raise ValueError('This independent replay is development-only')
scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
market=[]
for s in scenarios:
    infl=1.
    for o in s.observations:
        infl*=1+o.inflation
        market.append(dict(scenario_id=s.scenario_id,period=o.period,inflation_product=infl,
                           equity_return=o.nominal_returns[0],bond_return=o.nominal_returns[1],bill_return=o.nominal_returns[2]))
r=pd.read_csv(out/'decisions.csv.gz')
joined=r.merge(pd.DataFrame(market),on=['scenario_id','period'],how='left',validate='many_to_one')
assert len(joined)==len(r) and not joined.inflation_product.isna().any()
max_error=0.
for asset in ['equity','bond','bill']:
    expected=joined['post_trade_holdings_'+asset]*(1+joined[asset+'_return'])
    actual=joined['end_holdings_'+asset]
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-9)
    max_error=max(max_error,float(np.max(np.abs(actual-expected))))
np.testing.assert_allclose(joined.cumulative_inflation,joined.inflation_product,rtol=1e-12)
metrics=pd.read_csv(out/'scenario_metrics.csv')
ref=metrics[metrics.policy=='equal_annual'].set_index(['scenario_id','cost_rate','horizon'])
checks=0
for s in scenarios:
    for cost in sorted(ref.index.get_level_values('cost_rate').unique()):
        holdings=[0.,0.,100.]; infl=1.
        for o in s.observations:
            value=sum(holdings)
            transferred=sum(abs(value/3-h) for h in holdings)/2
            after=(value-cost*transferred)/3
            holdings=[after*(1+x) for x in o.nominal_returns]
            infl*=1+o.inflation
            key=(s.scenario_id,cost,o.period)
            if key in ref.index:
                actual=ref.loc[key].terminal_real_wealth
                assert abs(sum(holdings)/infl-actual)<1e-8
                checks+=1
# Actual average beginning-of-return-period equity exposure, not announced target.
weights=r[[f'post_trade_holdings_{a}' for a in ['equity','bond','bill']]].to_numpy()
weights=weights/weights.sum(axis=1,keepdims=True)
for i,a in enumerate(['equity','bond','bill']): r[a+'_exposure']=weights[:,i]
r.groupby(['policy','cost_rate']+(['evaluation_horizon'] if 'evaluation_horizon' in r else []))[
    ['equity_exposure','bond_exposure','bill_exposure']].mean().to_csv(out/'allocation_exposure.csv')
result=dict(all_checks_passed=True,market_rows_replayed=len(r),max_holding_discrepancy=max_error,
            independently_computed_reference_metrics=checks,holdout_used=False,
            scope='All saved return transitions and cumulative inflation; scalar independent equal-weight recurrence for every cost/horizon/scenario.')
(out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if json.loads((out/'config.json').read_text()).get('study_kind')=='scenario_efficiency':
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_efficiency.py'),str(out)],check=True)
if json.loads((out/'config.json').read_text()).get('study_kind')=='counterfactual_fqi':
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_learning.py'),str(out)],check=True)
if json.loads((out/'config.json').read_text()).get('study_kind')=='neural_path_control':
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_neural.py'),str(out)],check=True)
