#!/usr/bin/env python3
"""Scenario-separated diagnostic of a fixed, causal lag-feature forecaster."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'))
import numpy as np
import pandas as pd
from safebench.data import load_partition, folds
from safebench.experts import history_features


def diagnose(output):
    config=json.loads((ROOT/'configs/benchmark.json').read_text())
    scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    x=[]; y=[]; ids=[]
    for s in scenarios:
        for t in range(1,30):
            x.append(history_features(s.observations[:t]))
            o=s.observations[t]
            y.append([(1+r)/(1+o.inflation)-1 for r in o.nominal_returns])
            ids.append(s.scenario_id)
    x=np.array(x); y=np.array(y); ids=np.array(ids)
    rows=[]
    for fold,valid in enumerate(folds(range(1,121),5,config['seed'])):
        training=~np.isin(ids,valid); testing=~training
        mean=x[training].mean(axis=0); scale=np.maximum(x[training].std(axis=0),1e-3)
        z=np.column_stack((np.ones(len(x)),(x-mean)/scale))
        ridge=np.eye(z.shape[1]); ridge[0,0]=0
        beta=np.linalg.solve(z[training].T@z[training]+ridge,z[training].T@y[training])
        prediction=z[testing]@beta; reference=y[training].mean(axis=0)
        for j,asset in enumerate(['equity','bond','bill']):
            base=float(np.mean((y[testing,j]-reference[j])**2))
            loss=float(np.mean((y[testing,j]-prediction[:,j])**2))
            rows.append(dict(fold=fold,asset=asset,reference_mse=base,ridge_mse=loss,
                relative_mse_improvement=1-loss/base if base else 0.,validation_scenarios=len(valid),
                note='固定ridge=1；4项过去五年特征；整情景隔离；不等同投资收益改进或新的独立证据。'))
    pd.DataFrame(rows).to_csv(output,index=False)
    print(pd.DataFrame(rows).groupby('asset')[['reference_mse','ridge_mse','relative_mse_improvement']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description='仅开发集的固定因果特征预测诊断。')
    p.add_argument('--out',type=Path,required=True); a=p.parse_args(); diagnose(a.out)
