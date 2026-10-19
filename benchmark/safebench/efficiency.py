"""Nested selection and predeclared efficiency checks for scenario-utility policies."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .scenario_utility import CE_POLICY_NAMES


def validate_efficiency_config(config):
    if config.get('study_kind') != 'scenario_efficiency' or config.get('ce_valuation_horizons') != [1, 3]:
        raise ValueError('Expected one-year and three-year efficiency study')
    if config.get('efficiency_reference') != 'balanced_annual':
        raise ValueError('Efficiency requires the matched annual Balanced reference')
    for key in ('ce_gamma', 'efficiency_growth_margin', 'efficiency_p10_margin',
                'efficiency_p90_margin', 'efficiency_cost_margin'):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Invalid efficiency setting: ' + key)
    if not 0 < config['efficiency_trade_ratio'] < 1 or not 0 < config['review_drawdown'] < 1:
        raise ValueError('Invalid efficiency ratio/review threshold')
    if not any(np.isclose(cost, 2 * config['cost_rate']) for cost in config['sensitivity_costs']):
        raise ValueError('Doubled cost is required')
    grid = config.get('ce_tuning_grid', [])
    if not grid or len({(p['band'], p['buffer']) for p in grid}) != len(grid):
        raise ValueError('Empty or duplicate utility grid')
    for params in grid:
        if (not np.isfinite(params['band']) or not 0 <= params['band'] < 1
                or not np.isfinite(params['buffer']) or params['buffer'] < 0):
            raise ValueError('Invalid utility trial')


def _statistics(group):
    if group.empty or group.scenario_id.duplicated().any():
        raise ValueError('Empty/duplicated validation scenarios')
    values = group[['real_cagr', 'real_max_drawdown', 'trade_rate', 'total_real_cost']]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite validation metrics')
    return dict(median_growth=float(group.real_cagr.median()), p10_growth=float(group.real_cagr.quantile(.1)),
        p90_loss=float(group.real_max_drawdown.quantile(.9)), mean_trade_rate=float(group.trade_rate.mean()),
        mean_real_cost=float(group.total_real_cost.mean()))


def select_efficiency_trials(scores, reference, config):
    validate_efficiency_config(config)
    ref = _statistics(reference)
    if set(reference.policy) != {config['efficiency_reference']}:
        raise ValueError('Wrong efficiency reference')
    chosen, rows = {}, []
    expected_trials = set(range(len(config['ce_tuning_grid'])))
    if set(scores.policy) != set(CE_POLICY_NAMES):
        raise ValueError('Missing utility family')
    for name, family in scores.groupby('policy'):
        if set(family.trial) != expected_trials:
            raise ValueError('Incomplete utility trials')
        local = []
        for trial, group in family.groupby('trial'):
            if set(group.scenario_id) != set(reference.scenario_id):
                raise ValueError('Unpaired efficiency reference')
            stats = _statistics(group)
            dg = stats['median_growth'] - ref['median_growth']
            dp10 = stats['p10_growth'] - ref['p10_growth']
            dd = stats['p90_loss'] - ref['p90_loss']
            dt = stats['mean_trade_rate'] - config['efficiency_trade_ratio'] * ref['mean_trade_rate']
            df = stats['mean_real_cost'] - ref['mean_real_cost']
            checks = dict(growth_ok=dg >= -config['efficiency_growth_margin'] - 1e-12,
                p10_ok=dp10 >= -config['efficiency_p10_margin'] - 1e-12,
                p90_ok=dd <= config['efficiency_p90_margin'] + 1e-12,
                trades_ok=dt <= 1e-12, cost_ok=df <= config['efficiency_cost_margin'] + 1e-12)
            violation = max(0., (-config['efficiency_growth_margin'] - dg) / max(config['efficiency_growth_margin'], 1e-12),
                (-config['efficiency_p10_margin'] - dp10) / max(config['efficiency_p10_margin'], 1e-12),
                (dd - config['efficiency_p90_margin']) / max(config['efficiency_p90_margin'], 1e-12),
                dt / max(config['efficiency_trade_ratio'] * ref['mean_trade_rate'], 1e-12),
                (df - config['efficiency_cost_margin']) / max(ref['mean_real_cost'], 1e-12))
            local.append(dict(policy=name, trial=int(trial), **stats, **checks, risk_feasible=all(checks.values()),
                growth_difference=dg, p10_difference=dp10, p90_difference=dd,
                trade_constraint_difference=dt, cost_difference=df, maximum_violation=violation,
                reference_median_growth=ref['median_growth'], reference_p10_growth=ref['p10_growth'],
                reference_p90_loss=ref['p90_loss'], reference_trade_rate=ref['mean_trade_rate'],
                reference_real_cost=ref['mean_real_cost']))
        feasible = [row for row in local if row['risk_feasible']]
        status = 'feasible' if feasible else 'infeasible_diagnostic'
        # Infeasible families remain in every outer evaluation; they cannot freeze.
        pool = feasible or local
        selected = min(pool, key=lambda row: (
            0 if feasible else row['maximum_violation'], row['mean_trade_rate'],
            row['mean_real_cost'], -row['median_growth'], row['trial']))['trial']
        chosen[name] = dict(config['ce_tuning_grid'][selected], selection_trial=selected, selection_status=status)
        rows.extend(dict(row, selected=row['trial'] == selected, selection_status=status,
                         selection_rule='scenario_efficiency') for row in local)
    return chosen, rows


def tune_efficiency(scenarios, config, seed, trace_path=None):
    from .data import folds
    from .experiment import evaluate, dump, digest
    from .policies import fit_model
    validate_efficiency_config(config)
    if trace_path is not None:
        trace_path = Path(trace_path)
        if trace_path.exists():
            raise FileExistsError(trace_path)
    by_id = {s.scenario_id: s for s in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError('Duplicate tuning scenarios')
    collected, references, audits = [], [], []
    for inner, valid_ids in enumerate(folds(by_id, config['inner_folds'], seed)):
        train = [s for sid, s in by_id.items() if sid not in valid_ids]
        valid = [by_id[sid] for sid in valid_ids]
        started = time.perf_counter()
        model = fit_model(train, config)
        audits.append(dict(inner_fold=inner, train_ids=model['train_ids'], validation_ids=valid_ids,
            fit_seconds=time.perf_counter() - started, ce_blocks_sha256=digest(model['ce_blocks']),
            block_counts={key: len(value['gross']) for key, value in model['ce_blocks'].items()}))
        if trace_path is not None:
            directory = trace_path.parent.parent / 'tuning_models'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (trace_path.name.split('.')[0] + f'_inner_{inner}.json')
            if path.exists():
                raise FileExistsError(path)
            dump(path, model)
        ref, _ = evaluate(valid, model, {}, config, names=[config['efficiency_reference']],
                          horizons=[30], fold=inner, keep_records=False)
        ref['trial'] = -1
        references.append(ref)
        for trial, params in enumerate(config['ce_tuning_grid']):
            metrics, _ = evaluate(valid, model, {name: params for name in CE_POLICY_NAMES}, config,
                names=list(CE_POLICY_NAMES), horizons=[30], fold=inner, keep_records=False)
            metrics['trial'] = trial
            collected.append(metrics)
    scores = pd.concat(collected, ignore_index=True)
    reference = pd.concat(references, ignore_index=True)
    chosen, trials = select_efficiency_trials(scores, reference, config)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([scores, reference], ignore_index=True).to_csv(trace_path, index=False,
            compression={'method': 'gzip', 'mtime': 0})
    return chosen, trials, audits


def efficiency_intervals(metrics, config):
    """Family adjustment covers 2 policies x 5 endpoints x 2 required cost strata."""
    rng = np.random.default_rng(config['seed'])
    samples = config['efficiency_bootstrap']
    if not isinstance(samples, int) or samples < 20:
        raise ValueError('Invalid bootstrap budget')
    costs = [config['cost_rate'], 2 * config['cost_rate']]
    family_count = len(CE_POLICY_NAMES) * 5 * len(costs)
    tail = .05 / (2 * family_count)
    rows = []
    for cost in costs:
        block = metrics[(metrics.horizon == 30) & np.isclose(metrics.cost_rate, cost)]
        ref = block[block.policy == config['efficiency_reference']].set_index('scenario_id').sort_index()
        if ref.empty or ref.index.duplicated().any():
            raise ValueError('Missing/duplicated efficiency reference')
        indices = rng.integers(0, len(ref), (samples, len(ref)))
        for name in CE_POLICY_NAMES:
            group = block[block.policy == name].set_index('scenario_id').sort_index()
            if not group.index.equals(ref.index) or group.index.duplicated().any():
                raise ValueError('Unpaired efficiency metrics')
            for col, q, label, threshold, side in (
                ('real_cagr', .5, 'median_growth', -config['efficiency_growth_margin'], 'lower'),
                ('real_cagr', .1, 'p10_growth', -config['efficiency_p10_margin'], 'lower'),
                ('real_max_drawdown', .9, 'p90_loss', config['efficiency_p90_margin'], 'upper'),
                ('trade_rate', None, 'trade_constraint', 0., 'upper'),
                ('total_real_cost', None, 'mean_real_cost', config['efficiency_cost_margin'], 'upper')):
                a, b = group[col].to_numpy(), ref[col].to_numpy()
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Nonfinite efficiency endpoint')
                multiplier = config['efficiency_trade_ratio'] if label == 'trade_constraint' else 1.
                if q is None:
                    difference = float(a.mean() - multiplier * b.mean())
                    draws = a[indices].mean(axis=1) - multiplier * b[indices].mean(axis=1)
                else:
                    difference = float(np.quantile(a, q) - np.quantile(b, q))
                    draws = np.quantile(a[indices], q, axis=1) - np.quantile(b[indices], q, axis=1)
                lower, upper = float(np.quantile(draws, tail)), float(np.quantile(draws, 1-tail))
                passed = lower >= threshold - 1e-12 if side == 'lower' else upper <= threshold + 1e-12
                rows.append(dict(policy=name, reference=config['efficiency_reference'], horizon=30,
                    cost_rate=cost, metric=label, n=len(ref), difference=difference,
                    lower95=float(np.quantile(draws, .025)), upper95=float(np.quantile(draws, .975)),
                    family_lower=lower, family_upper=upper, threshold=threshold, bound=side,
                    passed=bool(passed), family_comparisons=family_count, bootstrap=samples))
    return pd.DataFrame(rows)


def candidate_suitability(metrics, profiles, config):
    from .metrics import binomial_upper
    rows = []
    # Three original portfolios plus two alternative Balanced policies.
    confidence = 1 - (1 - config['suitability_confidence']) / 5
    for profile in profiles:
        for name in CE_POLICY_NAMES:
            group = metrics[(metrics.policy == name) & (metrics.horizon == profile['horizon']) &
                            np.isclose(metrics.cost_rate, config['cost_rate'])]
            if group.empty or group.scenario_id.duplicated().any():
                raise ValueError('Missing/duplicated candidate customer horizon')
            breaches = int((group.real_max_drawdown > profile['loss_tolerance']).sum())
            upper = binomial_upper(breaches, len(group), confidence)
            liquidity_ok = profile['liquidity'] != 'high'
            rows.append(dict(profile=profile['profile_id'], policy=name, horizon=profile['horizon'],
                tolerance=profile['loss_tolerance'], n=len(group), breaches=breaches,
                breach_probability=breaches / len(group), upper_bound=upper, confidence=confidence,
                liquidity_ok=liquidity_ok, risk_pass=upper <= config['suitability_max_breach_probability'],
                eligible=bool(upper <= config['suitability_max_breach_probability'] and liquidity_ok),
                median_growth=float(group.real_cagr.median()), diagnostic_only=True))
    return pd.DataFrame(rows)


def write_efficiency_analysis(out, config):
    from . import ROOT
    from .metrics import paired_comparisons
    out = Path(out)
    metrics = pd.read_csv(out / 'scenario_metrics.csv')
    intervals = efficiency_intervals(metrics, config)
    intervals.to_csv(out / 'efficiency_intervals.csv', index=False)
    trials = pd.read_csv(out / 'tuning_trials.csv')
    screens = []
    for name in CE_POLICY_NAMES:
        chosen = trials[(trials.policy == name) & trials.selected]
        checks = dict(inner_selection_feasible=bool(len(chosen) == config['outer_folds'] and chosen.risk_feasible.all()))
        for cost_label, cost in (('official', config['cost_rate']), ('doubled', 2 * config['cost_rate'])):
            for row in intervals[(intervals.policy == name) & np.isclose(intervals.cost_rate, cost)].itertuples():
                checks[cost_label + '_' + row.metric] = bool(row.passed)
        screens.append(dict(policy=name, **checks, all_checks_passed=all(checks.values()),
            status='efficiency_established_development' if all(checks.values()) else 'not_established'))
    pd.DataFrame(screens).to_csv(out / 'efficiency_screen.csv', index=False)
    profiles = json.loads((ROOT / 'configs/profiles.json').read_text(encoding='utf-8'))
    candidate_suitability(metrics, profiles, config).to_csv(out / 'candidate_suitability.csv', index=False)
    names = ['balanced_annual_review', *CE_POLICY_NAMES]
    matched = metrics[(metrics.horizon == 30) & np.isclose(metrics.cost_rate, config['cost_rate']) & metrics.policy.isin(names)]
    paired_comparisons(matched, config['seed'], config['bootstrap'], baseline='balanced_annual_review').to_csv(
        out / 'review_matched_comparisons.csv', index=False)


def generate_efficiency_report(out, config):
    from .report import plt, LABELS
    out = Path(out)
    figures = out / 'figures'
    intervals = pd.read_csv(out / 'efficiency_intervals.csv')
    summary = pd.read_csv(out / 'summary.csv')
    trials = pd.read_csv(out / 'tuning_trials.csv')
    screen = pd.read_csv(out / 'efficiency_screen.csv')
    main = summary[(summary.horizon == 30) & np.isclose(summary.cost_rate, config['cost_rate'])].set_index('policy')

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(figures / (name + '.' + extension), dpi=170, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    official = intervals[np.isclose(intervals.cost_rate, config['cost_rate'])]
    for axis, endpoint, label in zip(axes, ('median_growth', 'p10_growth', 'p90_loss'),
        ('Median real CAGR difference\n(pp / year)', 'P10 real CAGR difference\n(pp / year)', 'P90 drawdown difference\n(pp)')):
        group = official[official.metric == endpoint].set_index('policy')
        for y, name in enumerate(CE_POLICY_NAMES):
            row = group.loc[name]
            axis.plot(100 * np.array([row.family_lower, row.family_upper]), [y, y], color='#287c8e', lw=2)
            axis.scatter(100 * row.difference, y, color='#173e5b')
        axis.axvline(100 * group.iloc[0].threshold, color='#ae5b32', ls='--')
        axis.axvline(0, color='gray', ls=':', lw=.8)
        axis.set_xlabel(label)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(2), [LABELS[name] for name in CE_POLICY_NAMES])
    save(fig, 'efficiency_intervals')

    names = ['balanced_annual', 'balanced_annual_review', 'balanced_threshold', 'balanced_gate', *CE_POLICY_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, col, scale, label in zip(axes, ('mean_trade_rate', 'mean_real_cost'), (100, 1),
        ('Post-initial reviews with trades (%)', 'Mean cumulative real transaction cost')):
        axis.barh(range(len(names)), scale * main.loc[names, col].to_numpy(), color='#287c8e')
        threshold = main.loc['balanced_annual', col] * (config['efficiency_trade_ratio'] if col == 'mean_trade_rate' else 1)
        axis.axvline(scale * threshold, color='#ae5b32', ls='--')
        axis.set_xlabel(label)
    axes[0].set_yticks(range(len(names)), [LABELS[name] for name in names])
    save(fig, 'efficiency_trading_cost')

    selected = trials[trials.selected].copy()
    fig, axis = plt.subplots(figsize=(8, 3))
    matrix = np.array([[config['ce_tuning_grid'][int(selected[(selected.policy == name) &
        (selected.outer_fold == fold)].iloc[0].trial)]['band'] for fold in range(config['outer_folds'])] for name in CE_POLICY_NAMES])
    axis.imshow(matrix, cmap='Blues', vmin=0, vmax=.1, aspect='auto')
    for y, name in enumerate(CE_POLICY_NAMES):
        for fold in range(config['outer_folds']):
            row = selected[(selected.policy == name) & (selected.outer_fold == fold)].iloc[0]
            axis.text(fold, y, f'{100*matrix[y,fold]:.0f}pp' + ('\nDIAGNOSTIC' if not row.risk_feasible else '\nFEASIBLE'),
                      ha='center', va='center', fontsize=9)
    axis.set_yticks(range(2), [LABELS[name] for name in CE_POLICY_NAMES])
    axis.set_xticks(range(config['outer_folds']), [str(i + 1) for i in range(config['outer_folds'])])
    axis.set_xlabel('Outer scenario fold')
    save(fig, 'efficiency_selected_bands')

    zh = {'utility_band_1y': '一年情景效用门控', 'utility_band_3y': '三年情景效用门控'}
    display = screen.copy()
    display['policy'] = display.policy.map(zh)
    display['status'] = display.status.map({'not_established': '未建立效率优势',
        'efficiency_established_development': '开发效率筛选通过'})
    display = display.rename(columns={'policy':'策略', 'inner_selection_feasible':'全部内层选择可行',
        'all_checks_passed':'全部条件通过', 'status':'状态',
        **{prefix + '_' + endpoint: label + meaning for prefix, label in (('official','原费用：'), ('doubled','双倍费用：'))
           for endpoint, meaning in (('median_growth','中位增长'), ('p10_growth','P10增长'),
               ('p90_loss','P90回撤'), ('trade_constraint','交易率'), ('mean_real_cost','实际费用'))}})
    diagnostics = selected[['outer_fold', 'policy', 'trial', 'risk_feasible', 'growth_difference',
                            'p10_difference', 'p90_difference', 'mean_trade_rate', 'selection_status']].copy()
    diagnostics['policy'] = diagnostics.policy.map(zh)
    diagnostics['selection_status'] = diagnostics.selection_status.map({'feasible':'可行', 'infeasible_diagnostic':'不可行诊断'})
    diagnostics = diagnostics.rename(columns={'outer_fold':'外层折', 'policy':'策略', 'trial':'参数编号',
        'risk_feasible':'内层条件通过', 'growth_difference':'中位增长差', 'p10_difference':'P10增长差',
        'p90_difference':'P90回撤差', 'mean_trade_rate':'交易率', 'selection_status':'选择状态'})
    body = ('<h1>固定目标与情景效用：交易效率研究</h1><p>固定Balanced为50/35/15；一年、三年'
        '估值各搜索六组漂移带/缓冲。仅使用开发情景1–120，5外折/3内折，共180次候选/内折评价。</p>'
        '<p>交易带外仍须通过效益/费用门槛。25%已观察实际回撤触发人工复核，等待期间保持持仓。'
        '年度复核不能保证避免损失，复核本身也不是止损；相同复核规则的年度参照用于识别暂停交易的影响。</p>'
        '<h2>预声明效率筛选</h2><p>相对年度Balanced，中位/P10年增长最多降低0.10个百分点，'
        'P90回撤最多增加1个百分点，交易率至少降低50%，实际费用不增加。原费用及双倍费用均须通过。'
        '区间按完整情景配对bootstrap，调整覆盖2策略×5端点×2费用的20项比较。区间条件于已拟合OOF模型，'
        '不包含重拟合、重叠训练折及研究迭代的不确定性；不是独立最终证据。</p>' + display.to_html(index=False) +
        '<h2>增长与风险差异</h2><img src="figures/efficiency_intervals.png">'
        '<p>线段为20项比较调整区间，棕色虚线为允许界限；单位为百分点。全部原始区间见efficiency_intervals.csv。</p>'
        '<h2>交易与费用</h2><img src="figures/efficiency_trading_cost.png">'
        '<p>虚线分别为年度参照的一半交易率与相同费用。原年度、相同复核年度、阈值及旧门控均为诊断参照。</p>'
        '<h2>选参状态</h2><img src="figures/efficiency_selected_bands.png">' + diagnostics.to_html(index=False) +
        '<p>无可行参数时按预声明最小最大标准化约束违反量选择诊断参数，仍评价全部情景并禁止冻结。</p>'
        '<h2>客户期限核验</h2><p>candidate_suitability.csv记录两候选在A–E对应期限的越界次数、'
        '对五个候选保护的一侧概率上界及流动性规则；这不自动修改既有客户分配，也不是损失保证。</p>'
        '<p><a href="report.html">完整基准、期限与严重情景报告</a>；逐情景指标、训练片段、选择和决策依据保存在同目录。'
        '三年CE是固定滚动终值估值代理，没有优化未来交易序列，不能解释为完整MPC或路径回撤保证。</p>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1"><title>SafeInvest 交易效率研究</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#183043}'
        'img{max-width:100%}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}'
        'td,th{padding:7px;border:1px solid #ccd6dc}</style><body>' + body + '</body></html>')
    (out / 'efficiency_report.html').write_text(document, encoding='utf-8')
    return out / 'efficiency_report.html'
