"""Independently rebuild utility blocks and replay saved utility/review decisions."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark'), str(ROOT / 'src')]
from safebench.data import load_partition


def verify(out):
    out = Path(out)
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['phase'] != 'development' or manifest['holdout_used']:
        raise PermissionError('Independent utility replay accepts development only')
    if config.get('study_kind') != 'scenario_efficiency':
        raise ValueError('Not a utility efficiency run')
    scenarios, partition_hash = load_partition(ROOT / 'data/SafeInvest_Synthetic_Data.csv')
    assert partition_hash == manifest['partition_sha256']
    by_id = {s.scenario_id: s for s in scenarios}
    market = {}
    for length in config['ce_valuation_horizons']:
        market[length] = {}
        for sid, scenario in by_id.items():
            rows = []
            for start in range(len(scenario.observations) - length + 1):
                observations = scenario.observations[start:start + length]
                denominator = np.prod([1 + o.inflation for o in observations])
                rows.append([np.prod([1 + o.nominal_returns[asset] for o in observations]) / denominator
                             for asset in range(3)])
            market[length][sid] = np.asarray(rows)
    fitted = json.loads((out / 'fitted_models.json').read_text())
    models = [row['model'] for row in fitted]
    models += [json.loads(path.read_text()) for path in sorted((out / 'tuning_models').glob('*.json'))]
    block_count, max_block_error = 0, 0.
    for model in models:
        ids = set(model['train_ids'])
        assert ids and ids <= set(range(1, 121))
        for key, block in model['ce_blocks'].items():
            length = int(key)
            assert block['horizon'] == length
            assert set(block['scenario_ids']) == ids
            assert len(block['gross']) == len(block['probabilities']) == len(block['start_periods']) == len(block['scenario_ids'])
            for sid in ids:
                positions = [i for i, value in enumerate(block['scenario_ids']) if value == sid]
                count = len(market[length][sid])
                assert [block['start_periods'][i] for i in positions] == list(range(1, count + 1))
                np.testing.assert_allclose(np.asarray(block['probabilities'])[positions], 1 / (len(ids) * count), rtol=1e-12)
            expected = np.array([market[length][sid][start - 1]
                for sid, start in zip(block['scenario_ids'], block['start_periods'])])
            observed = np.asarray(block['gross'])
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
            max_block_error = max(max_block_error, float(np.max(np.abs(observed - expected))))
            block_count += len(expected)
    records = pd.read_csv(out / 'decisions.csv.gz')
    anchor = np.array([.50, .35, .15])
    family_lengths = {'utility_band_1y': 1, 'utility_band_3y': 3}
    selected = records[records.policy.isin([*family_lengths, 'balanced_annual_review'])]
    cached = {}
    gamma = config['ce_gamma']

    def direct_ce(gross, probabilities, weights):
        payoff = gross @ weights
        assert np.isfinite(payoff).all() and (payoff > 0).all()
        if gamma == 1:
            return float(np.exp(np.sum(probabilities * np.log(payoff))))
        return float(np.sum(probabilities * payoff ** (1 - gamma)) ** (1 / (1 - gamma)))

    for row in fitted:
        for name, length in family_lengths.items():
            block = row['model']['ce_blocks'][str(length)]
            gross, probabilities = np.asarray(block['gross']), np.asarray(block['probabilities'])
            cached[(row['fold'], name)] = (gross, probabilities, direct_ce(gross, probabilities, anchor), row['params'][name])
    checked, gates, maximum_benefit_error = 0, 0, 0.
    for (name, sid, cost), group in selected.groupby(['policy', 'scenario_id', 'cost_rate']):
        peak, inflation, pending = config['initial_wealth'], 1., False
        for row in group.sort_values('period').itertuples():
            pre = np.array([getattr(row, 'pre_trade_holdings_' + asset) for asset in ('equity', 'bond', 'bill')])
            target = np.array([getattr(row, 'target_weights_' + asset) for asset in ('equity', 'bond', 'bill')])
            np.testing.assert_allclose(target, anchor, rtol=0, atol=1e-12)
            reasons = json.loads(row.reason_codes)
            if row.period == 1:
                assert row.action == 'rebalance' and reasons == ['INITIAL_ALLOCATION']
            elif pending:
                assert row.action == 'keep' and reasons == ['REVIEW_PENDING']
            elif 1 - pre.sum() / inflation / peak >= config['review_drawdown']:
                assert row.action == 'human_review' and reasons == ['DRAWDOWN_ALERT']
                pending = True
            elif name == 'balanced_annual_review':
                assert row.action == 'rebalance'
            else:
                gross, probabilities, target_ce, params = cached[(row.fold, name)]
                current = pre / pre.sum()
                if np.max(np.abs(current - anchor)) <= params['band']:
                    assert row.action == 'keep' and reasons == ['WITHIN_TRADE_BAND']
                else:
                    benefit = 1 - direct_ce(gross, probabilities, current) / target_ce
                    projected_fee = cost * np.abs(current - anchor).sum() / 2
                    error = abs(benefit - row.estimated_benefit)
                    assert error < 1e-10 and abs(projected_fee - row.estimated_cost) < 1e-10
                    maximum_benefit_error = max(maximum_benefit_error, error)
                    expected = 'rebalance' if benefit > projected_fee + params['buffer'] else 'keep'
                    assert row.action == expected
                    gates += 1
            assert bool(row.review_pending) == pending
            post = sum(getattr(row, 'post_trade_holdings_' + asset) for asset in ('equity', 'bond', 'bill'))
            peak = max(peak, post / inflation, row.real_wealth)
            inflation = row.cumulative_inflation
            checked += 1
    result = dict(all_checks_passed=True, holdout_used=False, training_models_checked=len(models),
        training_blocks_checked=block_count, maximum_block_error=max_block_error,
        decision_rows_checked=checked, utility_gates_checked=gates,
        maximum_benefit_error=maximum_benefit_error,
        verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Rebuild all inner/outer training blocks from development returns; independent power-form CE, band/cost gate, fixed targets and absorbing review replay.')
    destination = out / 'utility_replay.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != result:
            raise ValueError('Existing utility replay evidence differs; refusing to overwrite')
    else:
        with destination.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
