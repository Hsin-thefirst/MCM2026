"""Partition-first reader. Development never parses holdout return fields."""
import csv
import hashlib
import io
from itertools import islice
from pathlib import Path
import numpy as np
from safeinvest.contracts import MarketObservation, Scenario

COLUMNS = ('scenario_id','period','equity_return','bond_return','bill_proxy_return','inflation')

def load_partition(path, phase='development', authorized=False):
    if phase not in ('development', 'final'):
        raise ValueError('Unknown phase')
    if phase == 'final' and not authorized:
        raise PermissionError('Final data require a verified frozen bundle and exclusive evaluation lock')
    # Official CSV is ordered: verify exact keys, fail closed on reordering.
    # Only selected lines are decoded/numerically interpreted. No full-file hash.
    start, stop = (0,3600) if phase == 'development' else (3600,5400)
    with Path(path).open('rb') as f:
        header = next(f)
        raw = b''.join(islice(f, start, stop))
    reader = csv.DictReader(io.StringIO((header + raw).decode('utf-8-sig')))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError('Unexpected CSV schema')
    observations = []
    for index, row in enumerate(reader, start):
        sid, period = int(row['scenario_id']), int(row['period'])
        if (sid,period) != (index//30+1,index%30+1):
            raise ValueError('Expected ordered complete scenario grid; refusing partition ambiguity')
        vals = tuple(float(row[k]) for k in COLUMNS[2:])
        if not all(np.isfinite(v) and v > -1 for v in vals):
            raise ValueError('Invalid return/inflation')
        observations.append(MarketObservation(sid,period,vals[:3],vals[3]))
    if len(observations) != stop-start:
        raise ValueError('Incomplete partition')
    scenarios = tuple(Scenario(observations[i].scenario_id,tuple(observations[i:i+30]))
                      for i in range(0,len(observations),30))
    return scenarios, hashlib.sha256(header+raw).hexdigest()

def real_rows(scenarios):
    return np.array([[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns]
                     for s in scenarios for o in s.observations])

def folds(ids, k, seed):
    ids = np.asarray(sorted(ids), dtype=int)
    if len(set(ids)) != len(ids) or not 2 <= k <= len(ids):
        raise ValueError('Invalid folds')
    return [sorted(map(int,x)) for x in np.array_split(np.random.default_rng(seed).permutation(ids),k)]
