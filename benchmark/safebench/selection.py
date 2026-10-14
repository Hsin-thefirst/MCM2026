"""Inner-validation selection only; empirical path limits are not risk guarantees."""
import numpy as np


def validate_selection_config(config):
    rule = config.get('selection_rule', 'one_se_utility')
    if rule not in ('one_se_utility', 'path_risk'):
        raise ValueError('Unknown selection rule: ' + rule)
    if rule == 'path_risk':
        if config.get('selection_reference') not in ('balanced_annual', 'balanced_hold'):
            raise ValueError('Path-risk selection requires a matched Balanced reference')
        for key in ('selection_max_p90_increase', 'selection_max_p10_decrease'):
            value = config.get(key)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError('Invalid selection margin: ' + key)
        if config.get('selection_infeasible_action') != 'minimum_p90_diagnostic':
            raise ValueError('Infeasible selection must be an explicit diagnostic')
    return rule


def select_trials(scores, config, reference=None):
    """Return parameter dicts and complete trial diagnostics, preserving the legacy rule."""
    rule = validate_selection_config(config)
    scores = scores.copy()
    scores['utility'] = scores.real_cagr - config['selection_drawdown_penalty'] * scores.real_max_drawdown
    if rule == 'path_risk':
        if reference is None or reference.empty or reference.scenario_id.duplicated().any():
            raise ValueError('Missing or duplicated inner-validation reference')
        if set(reference.policy) != {config['selection_reference']}:
            raise ValueError('Wrong selection reference')
        ref_p90 = float(reference.real_max_drawdown.quantile(.9))
        ref_p10 = float(reference.real_cagr.quantile(.1))
    else:
        ref_p90 = ref_p10 = None
    rows, chosen = [], {}
    for name, group in scores.groupby('policy'):
        means = group.groupby('trial').utility.mean()
        if not np.isfinite(means).all():
            raise ValueError('Nonfinite trial utility')
        if set(means.index) != set(range(len(config['tuning_grid']))):
            raise ValueError('Incomplete trial set')
        pivot = group.pivot(index='scenario_id', columns='trial', values='utility')
        if pivot.isna().any().any():
            raise ValueError('Unpaired candidate validation scenarios')
        if rule == 'path_risk' and set(pivot.index) != set(reference.scenario_id):
            raise ValueError('Unpaired risk reference scenarios')
        p90 = group.groupby('trial').real_max_drawdown.quantile(.9)
        p10 = group.groupby('trial').real_cagr.quantile(.1)
        feasible = {int(i): rule != 'path_risk' or (
            p90[i] <= ref_p90 + config['selection_max_p90_increase'] + 1e-12 and
            p10[i] >= ref_p10 - config['selection_max_p10_decrease'] - 1e-12
        ) for i in means.index}
        admissible = [i for i in means.index if feasible[int(i)]]
        status = ('feasible' if admissible else 'infeasible_diagnostic') if rule == 'path_risk' else 'unconstrained'
        # Do not silently claim feasibility. Keep a diagnostic candidate so all
        # outer scenarios are still evaluated; freezing it is explicitly refused.
        if not admissible:
            minimum = float(p90.min())
            admissible = [i for i in means.index if p90[i] <= minimum + 1e-12]
        best = int(means.loc[admissible].idxmax())
        eligible, local_rows = [], []
        for trial in means.index:
            diff = pivot[best] - pivot[trial]
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            ok = trial in admissible and float(diff.mean()) <= se + 1e-12
            if ok:
                eligible.append(int(trial))
            local_rows.append(dict(policy=name, trial=int(trial), mean_utility=float(means[trial]),
                best_trial=best, paired_gap_se=se, one_se_eligible=bool(ok),
                p90_loss=float(p90[trial]), p10_growth=float(p10[trial]),
                reference_p90_loss=ref_p90, reference_p10_growth=ref_p10,
                risk_feasible=bool(feasible[int(trial)]) if rule == 'path_risk' else None, selection_rule=rule,
                selection_status=status))
        selected = max(eligible, key=lambda i: (
            config['tuning_grid'][i]['buffer'], config['tuning_grid'][i]['gamma'], -i))
        chosen[name] = dict(config['tuning_grid'][selected])
        if rule == 'path_risk':
            chosen[name]['selection_status'] = status
            chosen[name]['selection_trial'] = selected
        rows.extend(dict(row, selected=row['trial'] == selected) for row in local_rows)
    return chosen, rows
