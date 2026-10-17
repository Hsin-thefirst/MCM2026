"""Audit saved final evidence without reopening the market data or evaluating policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(run):
    out = Path(run)
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['phase'] != 'final':
        raise ValueError('This check consumes saved final outputs only')
    config = json.loads((out/'config.json').read_text())
    frozen = json.loads((out/'frozen_bundle.json').read_text())
    records = pd.read_csv(out/'decisions.csv.gz')
    metrics = pd.read_csv(out/'scenario_metrics.csv')
    costs = [config['cost_rate']]+config['sensitivity_costs']
    assets = ('equity', 'bond', 'bill')
    ref = records[(records.policy == 'equal_annual')&
                  np.isclose(records.cost_rate, config['cost_rate'])&
                  (records.simulation_horizon == max(config['horizons']))].copy()
    if ref.duplicated(['scenario_id', 'period']).any():
        raise ValueError('Ambiguous saved market reference')
    recovered = ref[['scenario_id', 'period', 'cumulative_inflation']].copy()
    for a in assets:
        recovered[a+'_return'] = ref['end_holdings_'+a]/ref['post_trade_holdings_'+a]-1
    joined = records.merge(recovered, on=['scenario_id', 'period'], validate='many_to_one',
                           suffixes=('', '_reference'), how='left')
    if joined.equity_return.isna().any():
        raise ValueError('Incomplete saved reference paths')
    error = 0.
    for a in assets:
        expected = joined['post_trade_holdings_'+a]*(1+joined[a+'_return'])
        np.testing.assert_allclose(joined['end_holdings_'+a], expected, rtol=1e-12, atol=1e-9)
        error = max(error, float(np.max(np.abs(joined['end_holdings_'+a]-expected))))
    np.testing.assert_allclose(joined.cumulative_inflation, joined.cumulative_inflation_reference, rtol=1e-12)
    reference_metrics = metrics[metrics.policy == 'equal_annual'].set_index(['scenario_id', 'cost_rate', 'horizon'])
    checks = 0
    for sid, path in recovered.groupby('scenario_id'):
        for cost in costs:
            holdings = [0., 0., config['initial_wealth']]
            initial = peak = config['initial_wealth']
            drawdown = real_cost = 0.; previous_inflation = 1.
            for row in path.sort_values('period').itertuples():
                wealth = sum(holdings)
                moved = sum(abs(wealth/3-v) for v in holdings)/2
                fee = cost*moved
                real_cost += fee/previous_inflation
                post = (wealth-fee)/previous_inflation
                drawdown = max(drawdown, 1-post/peak); peak = max(peak, post)
                holdings = [(wealth-fee)/3*(1+getattr(row, a+'_return')) for a in assets]
                real = sum(holdings)/row.cumulative_inflation
                drawdown = max(drawdown, 1-real/peak); peak = max(peak, real)
                previous_inflation = row.cumulative_inflation
                key = (sid, cost, row.period)
                if key in reference_metrics.index:
                    saved = reference_metrics.loc[key]
                    np.testing.assert_allclose([real, (real/initial)**(1/row.period)-1, drawdown, real_cost],
                        [saved.terminal_real_wealth, saved.real_cagr, saved.real_max_drawdown, saved.total_real_cost],
                        rtol=1e-12, atol=1e-9)
                    checks += 1
    assigned = pd.read_csv(out/'suitability.csv').set_index('profile_id')
    for profile in frozen['frozen_suitability']:
        saved = assigned.loc[profile['profile_id']]
        assert saved.portfolio == profile['portfolio'] and saved.policy == profile['policy']
        if profile['portfolio'] != 'human_review':
            group = metrics[(metrics.policy == profile['policy'])&(metrics.horizon == profile['horizon'])&
                            np.isclose(metrics.cost_rate, config['cost_rate'])]
            np.testing.assert_allclose(saved.final_median_growth, group.real_cagr.median(), atol=1e-13)
            np.testing.assert_allclose(saved.final_breach_probability,
                (group.real_max_drawdown > profile['loss_tolerance']).mean(), atol=1e-13)
    result = dict(all_checks_passed=True, market_data_reopened=False, policies_reevaluated=False,
        saved_transitions_checked=len(records), cross_policy_return_max_error=error,
        scalar_reference_checks=checks, frozen_assignments_checked=len(assigned),
        scope='保存账本之间的市场一致性、独立等权核算及冻结客户分配；不重新访问最终市场数据。')
    with (out/'saved_final_verification.json').open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run'); verify(parser.parse_args().run)
