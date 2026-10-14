import numpy as np
from scipy.stats import beta

def compute_metrics(result):
    records, cfg = result.records, result.config
    annual = np.array([cfg.initial_wealth]+[r.real_wealth for r in records])
    returns = annual[1:]/annual[:-1]-1
    marks = [cfg.initial_wealth]
    inflation = 1.
    real_cost = 0.
    turnover = []
    for r in records:
        marks.extend([sum(r.post_trade_holdings)/inflation,r.real_wealth])
        real_cost += r.transaction_cost/inflation
        turnover.append(r.moved_amount/sum(r.pre_trade_holdings))
        inflation = r.cumulative_inflation
    marks = np.array(marks)
    mdd = float(np.max(1-marks/np.maximum.accumulate(marks)))
    trades = [r.moved_amount > 1e-10*cfg.initial_wealth for r in records]
    return dict(scenario_id=result.scenario_id,policy=result.policy_name,horizon=cfg.horizon,
        cost_rate=cfg.transaction_cost_rate,real_cagr=(annual[-1]/annual[0])**(1/cfg.horizon)-1,
        real_max_drawdown=mdd,real_return_volatility=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.,
        terminal_real_wealth=annual[-1],trade_rate=float(np.mean(trades[1:])) if len(trades)>1 else 0.,
        trade_rate_including_initial=float(np.mean(trades)),total_nominal_cost=sum(r.transaction_cost for r in records),
        total_real_cost=real_cost,mean_turnover=float(np.mean(turnover)),worst_real_return=float(min(returns)),
        annual_return_cvar90=expected_shortfall(-returns),
        terminal_loss=float(annual[-1]<annual[0]),review=float(any(r.review_pending for r in records)),
        review_first_period=next((r.period for r in records if r.review_pending),0))

def expected_shortfall(values, alpha=.9):
    """Upper tail mean with fractional boundary mass; exact empirical CVaR."""
    x = np.sort(np.asarray(values,dtype=float))[::-1]
    if not len(x) or not 0 <= alpha < 1:
        raise ValueError('Invalid CVaR input')
    mass = len(x)*(1-alpha)
    k = int(np.floor(mass))
    return float((x[:k].sum()+(mass-k)*x[k] if k<len(x) else x.sum())/mass)

def summarize(frame):
    import pandas as pd
    rows=[]
    for keys,g in frame.groupby(['policy','horizon','cost_rate'],sort=True):
        if g.scenario_id.duplicated().any():
            raise ValueError('Duplicate scenario in aggregate')
        rows.append(dict(policy=keys[0],horizon=keys[1],cost_rate=keys[2],n=len(g),
            median_growth=g.real_cagr.median(),p10_growth=g.real_cagr.quantile(.1),
            median_loss=g.real_max_drawdown.median(),p90_loss=g.real_max_drawdown.quantile(.9),
            median_volatility=g.real_return_volatility.median(),mean_trade_rate=g.trade_rate.mean(),
            mean_trade_rate_including_initial=g.trade_rate_including_initial.mean(),
            mean_real_cost=g.total_real_cost.mean(),mean_turnover=g.mean_turnover.mean(),
            terminal_loss_probability=g.terminal_loss.mean(),cvar90_drawdown=expected_shortfall(g.real_max_drawdown),
            median_annual_return_cvar90=g.annual_return_cvar90.median(),
            review_rate=g.review.mean()))
    return pd.DataFrame(rows)

def binomial_upper(k,n,confidence=.95):
    """Exact one-sided Clopper-Pearson bound; iid scenarios assumed."""
    return 1. if k == n else float(beta.ppf(confidence,k+1,n-k))

def paired_comparisons(frame, seed=2026, bootstrap=2000, baseline='equal_annual'):
    import pandas as pd
    rng=np.random.default_rng(seed)
    rows=[]
    for (h,cost),block in frame.groupby(['horizon','cost_rate']):
        ref=block[block.policy==baseline].set_index('scenario_id').sort_index()
        others=[p for p in sorted(block.policy.unique()) if p != baseline]
        idx=rng.integers(0,len(ref),(bootstrap,len(ref)))
        # Bonferroni across policies for each predeclared metric/stratum; exploratory, conditional on fitted OOF models.
        tail=.05/(2*max(1,len(others)))
        for p in others:
            g=block[block.policy==p].set_index('scenario_id').sort_index()
            if not g.index.equals(ref.index):
                raise ValueError('Unpaired scenarios')
            for col,q,label in [('real_cagr',.5,'median_growth'),('real_cagr',.1,'p10_growth'),
                                ('real_max_drawdown',.5,'median_loss'),('real_max_drawdown',.9,'p90_loss')]:
                a,b=g[col].to_numpy(),ref[col].to_numpy()
                boot=np.quantile(a[idx],q,axis=1)-np.quantile(b[idx],q,axis=1)
                rows.append(dict(policy=p,horizon=h,cost_rate=cost,metric=label,
                    difference=float(np.quantile(a,q)-np.quantile(b,q)),
                    lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                    family_lower=float(np.quantile(boot,tail)),family_upper=float(np.quantile(boot,1-tail)),
                    paired_win_rate=float(np.mean(a>b)) if col=='real_cagr' else float(np.mean(a<b))))
    return pd.DataFrame(rows)

def pareto_flags(summary):
    result=summary.copy()
    result['pareto']=False
    for _,g in result.groupby(['horizon','cost_rate']):
        values=g[['median_growth','p10_growth','p90_loss','mean_real_cost']].to_numpy()*[1,1,-1,-1]
        for i,index in enumerate(g.index):
            result.loc[index,'pareto']=not any(np.all(v>=values[i]) and np.any(v>values[i]) for v in values)
    return result
