"""Paired ablation of inner-selection rules; never reads market observations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import TUNED, digest, dump


def compare_selection_runs(control, constrained, out):
    control, constrained, out = map(Path, (control, constrained, out))
    sources = [control, constrained]
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in sources]
    configs = [json.loads((p / 'config.json').read_text()) for p in sources]
    for manifest, config in zip(manifests, configs):
        if manifest['phase'] != 'development' or manifest.get('holdout_used'):
            raise PermissionError('Selection research accepts development results only')
        if manifest['config_sha256'] != digest(config):
            raise ValueError('Source configuration hash mismatch')
        if not manifest['scenario_ids'] or not set(manifest['scenario_ids']) <= set(range(1, 121)):
            raise PermissionError('Invalid development scenario IDs')
    for key in ('code_sha256', 'partition_sha256', 'scenario_ids', 'dependencies', 'python'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Study sources differ: ' + key)
    if configs[0].get('selection_rule') != 'one_se_utility' or configs[1].get('selection_rule') != 'path_risk':
        raise ValueError('Expected utility-control and path-risk selection')
    without_rule = [{k: v for k, v in c.items() if k != 'selection_rule'} for c in configs]
    if without_rule[0] != without_rule[1]:
        raise ValueError('Ablation may change the selection rule only, not its search budget')
    split_data = [json.loads((p / 'splits.json').read_text()) for p in sources]
    if split_data[0] != split_data[1]:
        raise ValueError('Unmatched nested scenario splits')
    frames = [pd.read_csv(p / 'scenario_metrics.csv') for p in sources]
    keys = ['policy', 'scenario_id', 'horizon', 'cost_rate']
    for frame in frames:
        if frame.duplicated(keys).any() or set(frame.scenario_id) != set(manifests[0]['scenario_ids']):
            raise ValueError('Incomplete or duplicated study metrics')
    indexed = [f.set_index(keys).sort_index() for f in frames]
    if not indexed[0].index.equals(indexed[1].index):
        raise ValueError('Unmatched study metrics')
    if not indexed[0]['fold'].equals(indexed[1]['fold']):
        raise ValueError('Unmatched outer folds')
    names = sorted(set(frames[0].policy) & set(TUNED))
    if not names:
        raise ValueError('No gated policies to compare')
    config = configs[1]
    rng = np.random.default_rng(config['seed'])
    rows = []
    endpoints = [('real_cagr', .5, 'median_growth'), ('real_cagr', .1, 'p10_growth'),
                 ('real_max_drawdown', .5, 'median_loss'), ('real_max_drawdown', .9, 'p90_loss')]
    for (horizon, cost), block in frames[0].groupby(['horizon', 'cost_rate']):
        base = block[block.policy == names[0]].set_index('scenario_id').sort_index()
        indices = rng.integers(0, len(base), (config['bootstrap'], len(base)))
        tail = .05 / (2 * len(names))
        for name in names:
            left = block[block.policy == name].set_index('scenario_id').sort_index()
            right = frames[1][(frames[1].policy == name) & (frames[1].horizon == horizon) &
                              np.isclose(frames[1].cost_rate, cost)].set_index('scenario_id').sort_index()
            if not left.index.equals(base.index) or not right.index.equals(base.index):
                raise ValueError('Unpaired scenario IDs')
            for col, q, label in endpoints:
                a, b = right[col].to_numpy(), left[col].to_numpy()
                samples = np.quantile(a[indices], q, axis=1) - np.quantile(b[indices], q, axis=1)
                rows.append(dict(policy=name, horizon=int(horizon), cost_rate=float(cost), metric=label,
                    n=len(base), difference=float(np.quantile(a, q) - np.quantile(b, q)),
                    lower95=float(np.quantile(samples, .025)), upper95=float(np.quantile(samples, .975)),
                    family_lower=float(np.quantile(samples, tail)), family_upper=float(np.quantile(samples, 1-tail)),
                    comparison_count=len(names)))
    result = pd.DataFrame(rows)
    summary = []
    for label, frame in zip(('utility_control', 'path_risk'), frames):
        for (name, horizon, cost), group in frame.groupby(['policy', 'horizon', 'cost_rate']):
            summary.append(dict(selection=label, policy=name, horizon=int(horizon), cost_rate=float(cost),
                median_growth=float(group.real_cagr.median()), p10_growth=float(group.real_cagr.quantile(.1)),
                median_loss=float(group.real_max_drawdown.median()), p90_loss=float(group.real_max_drawdown.quantile(.9)),
                mean_trade_rate=float(group.trade_rate.mean()), mean_real_cost=float(group.total_real_cost.mean())))
    out.mkdir(parents=True, exist_ok=False)
    result.to_csv(out / 'paired_selection.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'source_summary.csv', index=False)
    source_hashes = {str(p.resolve()): {
        name: hashlib.sha256((p / name).read_bytes()).hexdigest()
        for name in ('manifest.json', 'config.json', 'scenario_metrics.csv', 'splits.json')
    } for p in sources}
    dump(out / 'manifest.json', dict(phase='development_selection_ablation', holdout_used=False,
        source_sha256=source_hashes, seed=config['seed'], bootstrap=config['bootstrap'],
        comparison_count=len(names), difference='path_risk minus utility_control',
        uncertainty='Conditional paired scenario bootstrap; refitting and research-iteration uncertainty excluded.'))
    return result


def generate_selection_report(out, config):
    """Make source-backed Chinese HTML and caption-free vector comparison plots."""
    from .report import plt, LABELS
    out = Path(out)
    summary = pd.read_csv(out / 'source_summary.csv')
    pairs = pd.read_csv(out / 'paired_selection.csv')
    names = sorted(set(pairs.policy))
    main = summary[(summary.horizon == 30) & np.isclose(summary.cost_rate, config['cost_rate'])]
    main_pairs = pairs[(pairs.horizon == 30) & np.isclose(pairs.cost_rate, config['cost_rate'])]
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(figures / (name + '.' + extension), dpi=170, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        group = main[main.policy == name].set_index('selection')
        a, b = group.loc['utility_control'], group.loc['path_risk']
        ax.annotate('', (100*b.p90_loss, 100*b.median_growth),
                    (100*a.p90_loss, 100*a.median_growth), arrowprops=dict(arrowstyle='->', color=f'C{i}'))
        ax.scatter(100*a.p90_loss, 100*a.median_growth, marker='o', color=f'C{i}', s=50)
        ax.scatter(100*b.p90_loss, 100*b.median_growth, marker='D', color=f'C{i}', s=60, label=LABELS[name])
    ax.scatter([], [], marker='o', color='gray', label='Utility control')
    ax.scatter([], [], marker='D', color='gray', label='Path-risk selection')
    ax.set(xlabel='P90 real maximum drawdown (%)', ylabel='Median real CAGR (%)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'selection_tradeoff')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, metric, label in zip(axes, ('median_growth', 'p90_loss'),
                                    ('Median CAGR change (pp / year)', 'P90 drawdown change (pp)')):
        block = main_pairs[main_pairs.metric == metric].set_index('policy')
        for y, name in enumerate(names):
            row = block.loc[name]
            axis.plot(100*np.array([row.lower95, row.upper95]), [y, y], color='#287c8e', lw=2)
            axis.scatter(100*row.difference, y, color='#173e5b', s=35)
        axis.axvline(0, color='gray', ls='--')
        axis.set_xlabel(label)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(names)), [LABELS[n] for n in names])
    fig.tight_layout()
    save(fig, 'selection_intervals')

    display = main[main.policy.isin(names)].copy()
    display.selection = display.selection.map({'utility_control':'效用选择对照', 'path_risk':'路径风险约束选择'})
    zh = {'balanced_gate':'固定均衡成本门控', 'cost_aware':'成本感知配置', 'adaptive':'自适应配置',
          'adaptive_review':'自适应加人工复核'}
    display.policy = display.policy.map(zh)
    for col in ('median_growth', 'p10_growth', 'median_loss', 'p90_loss', 'mean_trade_rate'):
        display[col] = display[col].map(lambda value: f'{100*value:.2f}%')
    display = display.drop(columns=['horizon', 'cost_rate']).rename(columns={
        'selection':'选择规则', 'policy':'策略', 'median_growth':'中位实际CAGR', 'p10_growth':'P10增长',
        'median_loss':'中位回撤', 'p90_loss':'P90回撤', 'mean_trade_rate':'后续交易率', 'mean_real_cost':'平均实际费用'})
    body = ('<h1>路径风险约束的模型选择研究</h1><p>两个运行使用相同情景折、8组参数、'
        '初始配置、成本和17种策略，仅改变内层选择规则。约束参照为同初始Balanced年度再平衡，'
        'P90回撤允许高1个百分点、P10增长允许低0.10个百分点/年。约束只看内层折外路径。</p>'
        '<p>本研究重复使用开发情景1–120，没有读取最终留出集。区间条件于已拟合的折外程序，'
        '不包含重拟合和重复研究的不确定性。即使内层通过，也不保证外层或未来风险。</p>'
        '<h2>30年、0.25%费用下的配对结果</h2>' + display.to_html(index=False) +
        '<h2>增长与路径风险变化</h2><img src="figures/selection_tradeoff.png">'
        '<p>圆点为效用对照、菱形为风险约束选择，箭头连接同一策略族。</p>'
        '<h2>相对效用对照的变化</h2><img src="figures/selection_intervals.png">'
        '<p>线段为完整情景配对bootstrap的点态95%区间。四策略族调整区间及全部期限/成本见'
        ' paired_selection.csv；不同期限、成本和研究迭代不受该调整同时保护。</p>'
        '<p><a href="../control/report.html">效用对照详细报告</a> · '
        '<a href="../constrained/report.html">风险约束详细报告</a>。'
        '各自报告保留无可行候选的诊断状态、完整调参依据及晋级筛选。图表另存SVG和无标题PDF。</p>')
    (out / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>SafeInvest 路径风险研究</title><style>body{font:16px/1.6 system-ui;max-width:1150px;'
        'margin:32px auto;padding:0 20px;color:#183043}img{max-width:100%}table{border-collapse:collapse;'
        'font-size:13px}td,th{padding:8px;border:1px solid #ccd6dc}</style><body>' + body + '</body></html>', encoding='utf-8')
    return out / 'report.html'
