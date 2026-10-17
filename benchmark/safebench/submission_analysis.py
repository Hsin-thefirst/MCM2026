"""Saved-output comparisons and customer sensitivity; no additional fitting."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_comparisons
from .submission import BASE_RULES, POLICIES


def write_customer_sensitivity(out, metrics, config):
    from .experiment import suitability
    assignments, risks = [], []
    for balanced in BASE_RULES+['submission_selected']:
        for mode, floor in [('legacy', .4), ('strategic_bill_floor', .4),
                            ('strategic_bill_floor', .2), ('strategic_bill_floor', .3)]:
            cfg = copy.deepcopy(config)
            cfg['submission'].update(liquidity_mode=mode, liquidity_floor=floor)
            mapping = dict(conservative='plan_a_conservative', balanced=balanced, growth='plan_a_growth')
            a, r = suitability(metrics, cfg, mapping)
            a['balanced_candidate'] = balanced
            a['liquidity_mode'] = mode
            a['liquidity_floor'] = floor
            r['balanced_candidate'] = balanced
            r['liquidity_floor'] = floor
            assignments.append(a); risks.append(r)
    pd.concat(assignments, ignore_index=True).to_csv(Path(out)/'customer_rule_sensitivity.csv', index=False)
    pd.concat(risks, ignore_index=True).to_csv(Path(out)/'customer_candidate_risk.csv', index=False)


def comparisons_and_screen(out, config):
    out = Path(out)
    metrics = pd.read_csv(out/'scenario_metrics.csv')
    main = metrics[(metrics.horizon == 30)&np.isclose(metrics.cost_rate, config['cost_rate'])]
    high = metrics[(metrics.horizon == 30)&np.isclose(metrics.cost_rate, .005)]
    references = ['plan_a_balanced', 'submission_c_full', 'balanced_annual', 'plan_a_annual', 'equal_annual']
    tables = []
    for ref in references:
        table = paired_comparisons(main, config['seed'], config['bootstrap'], baseline=ref)
        table['reference'] = ref
        tables.append(table)
    pairs = pd.concat(tables, ignore_index=True)
    pairs.to_csv(out/'submission_comparisons.csv', index=False)
    fitted = json.loads((out/'fitted_models.json').read_text())
    settings = config['plan_a']['risk_promotion']
    rows = []
    for ref in references:
        candidates = ['submission_c_cap10', 'submission_c_cap20'] if ref == 'submission_c_full' else POLICIES
        for name in candidates:
            t = pairs[(pairs.policy == name)&(pairs.reference == ref)].set_index('metric')
            growth, p10, risk = [t.loc[k] for k in ('median_growth', 'p10_growth', 'p90_loss')]
            gc = main[main.policy == name]; gr = main[main.policy == ref]
            hc = high[high.policy == name]; hr = high[high.policy == ref]
            growth_folds = risk_folds = 0
            for fold in sorted(main.fold.unique()):
                c = gc[gc.fold == fold]; r = gr[gr.fold == fold]
                growth_folds += int(c.real_cagr.median() > r.real_cagr.median())
                risk_folds += int(c.real_max_drawdown.quantile(.9) < r.real_max_drawdown.quantile(.9))
            feasible = all(f['params'][name]['execution_selection_feasible'] for f in fitted)
            review_ok = bool(gc.review.mean() <= settings['maximum_review_rate'])
            gchecks = dict(practical_growth=bool(growth.difference >= config['minimum_growth_improvement']),
                growth_lower_positive=bool(growth.family_lower > 0), p10_noninferior=bool(p10.family_lower >= -.001),
                p90_noninferior=bool(risk.family_upper <= config['maximum_p90_loss_increase']),
                fold_stability=growth_folds >= 4, doubled_cost_growth=bool(hc.real_cagr.median() > hr.real_cagr.median()),
                inner_feasible=feasible, review_rate=review_ok)
            rchecks = dict(practical_risk=bool(risk.difference <= -settings['minimum_p90_reduction']),
                risk_upper_negative=bool(risk.family_upper < 0),
                growth_noninferior=bool(growth.family_lower >= -settings['maximum_growth_loss']),
                p10_noninferior=bool(p10.family_lower >= -settings['maximum_p10_loss']),
                fold_stability=risk_folds >= 4,
                doubled_cost=bool(hc.real_max_drawdown.quantile(.9) < hr.real_max_drawdown.quantile(.9) and
                    hc.real_cagr.median() >= hr.real_cagr.median()-settings['maximum_growth_loss']),
                inner_feasible=feasible, review_rate=review_ok)
            rows.append(dict(policy=name, reference=ref,
                primary=name == 'submission_selected' and ref == 'plan_a_balanced',
                growth_difference=growth.difference, p90_difference=risk.difference,
                growth_lower=growth.family_lower, growth_upper=growth.family_upper,
                p90_lower=risk.family_lower, p90_upper=risk.family_upper,
                positive_growth_folds=growth_folds, positive_risk_folds=risk_folds,
                growth_route=all(gchecks.values()), risk_route=all(rchecks.values()),
                growth_checks=json.dumps(gchecks), risk_checks=json.dumps(rchecks)))
    screen = pd.DataFrame(rows)
    screen.to_csv(out/'submission_screen.csv', index=False)
    selected = screen[screen.policy == 'submission_selected'].set_index('reference')
    primary_ok = bool(selected.loc['plan_a_balanced', ['growth_route', 'risk_route']].any())
    growth_route = bool(selected.loc['balanced_annual', 'growth_route'])
    risk_route = bool(selected.loc[['balanced_annual', 'plan_a_annual'], 'risk_route'].all())
    service = pd.read_csv(out/'suitability_risk.csv')
    order_ok = bool(service.risk_order_ok.all())
    adopted = primary_ok and (growth_route or risk_route) and order_ok
    decision = dict(selected_policy='submission_selected' if adopted else 'plan_a_balanced',
        adopted_new_execution=adopted, primary_screen=primary_ok,
        original_growth_route=growth_route, original_risk_route=risk_route,
        product_risk_order=order_ok, holdout_used=False,
        rule='Promote only the preregistered nested selector; otherwise retain calibrated A. No post-hoc choice of individual challengers.',
        note='Development screens are exploratory and do not guarantee final or live performance.')
    (out/'submission_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
