"""Training-only path-risk budgets and deterministic scenario-value experts.

The neural comparator distils a scenario planner; it is not labelled RL. Forecast
paths are training scenarios, never the current evaluation scenario's future.
"""
import time
import numpy as np
from safeinvest.contracts import Action, Decision
from .experts import Adam, gradients, init_network, pack_network, unpack_network
from .policies import grid, EQUAL, PORTFOLIOS

FRONTIER_NAMES = ('frontier_annual', 'frontier_hold', 'frontier_gate',
                  'frontier_mpc', 'frontier_linear', 'frontier_neural')
MATCHED_NAMES = ('frontier_matched_annual', 'frontier_matched_hold', 'frontier_matched_gate')
STATE_NAMES = ('equity', 'bond', 'bill', 'drawdown', 'maximum_drawdown',
               'remaining_scaled', 'horizon_scaled', 'remaining_fraction',
               'cost_rate', 'strategic_equity', 'strategic_bond')


def enabled(config):
    return config.get('frontier_research', {}).get('enabled', False)


def tail_mean(values, alpha=.9, axis=0):
    values = np.sort(values, axis=axis)
    values = np.moveaxis(values, axis, 0)[::-1]
    mass = len(values)*(1-alpha)
    k = int(np.floor(mass))
    return (values[:k].sum(axis=0)+(mass-k)*values[k])/mass


def annual_paths(real, targets, cost):
    """All targets x all complete training paths, including fee-time losses."""
    n, steps, _ = real.shape
    weights = np.broadcast_to(targets[:, None, :], (len(targets), n, 3)).copy()
    wealth = np.ones((len(targets), n))
    peak = wealth.copy(); loss = np.zeros_like(wealth)
    before = np.zeros_like(weights); before[:, :, 2] = 1
    growths = []; losses = []
    for t in range(steps):
        fee = cost*.5*np.abs(weights-before).sum(axis=2)
        after = wealth*(1-fee)
        loss = np.maximum(loss, 1-after/peak)
        end_assets = weights*(1+real[None, :, t, :])
        gross = end_assets.sum(axis=2)
        wealth = after*gross
        peak = np.maximum(peak, wealth)
        loss = np.maximum(loss, 1-wealth/peak)
        before = end_assets/gross[:, :, None]
        growths.append(wealth**(1/(t+1))-1); losses.append(loss.copy())
    return np.array(growths), np.array(losses)


def state_features(weights, drawdown, maximum, remaining, horizon, cost, strategic):
    n = len(weights)
    return np.column_stack((weights, drawdown, maximum, remaining/30, horizon/30,
                            remaining/horizon, np.broadcast_to(cost, n), strategic[:, :2]))


def value_estimates(targets, weights, drawdown, maximum, remaining, horizon,
                    gross_paths, penalty, cost, risk_barrier=0., risk_slack=.01):
    """Empirical log certainty-equivalent with tail mean of future path MDD.

    A candidate is bought now, then held until the horizon. This continuation is
    common to all actions. At the next real review the planner runs again.
    """
    fees = cost*.5*np.abs(targets-weights[:, None, :]).sum(axis=2)
    q = np.empty(fees.shape)
    for rem in np.unique(remaining):
        ids = np.flatnonzero(remaining == rem)
        future = np.einsum('bak,stk->bast', targets[ids], gross_paths[:, :int(rem)])
        future *= (1-fees[ids])[:, :, None, None]
        relative_peak = 1/np.maximum(1-drawdown[ids], 1e-9)
        peak = np.maximum.accumulate(future, axis=3)
        peak = np.maximum(peak, relative_peak[:, None, None, None])
        loss = np.maximum((1-future/peak).max(axis=3),
                          maximum[ids, None, None])
        fee_loss = 1-(1-fees[ids])*(1-drawdown[ids, None])
        loss = np.maximum(loss, fee_loss[:, :, None])
        risk = tail_mean(loss, axis=2)-maximum[ids, None]
        q[ids] = np.log(future[:, :, :, -1]).mean(axis=2)/horizon[ids, None]-penalty*risk
        if risk_barrier:
            p90=np.quantile(loss,.9,axis=2)
            q[ids]-=risk_barrier*np.maximum(0,p90-p90[:,1:2]-risk_slack)
    return q, fees


def network_values(network, x):
    cache = [x]
    for i, (w, b) in enumerate(zip(network['weights'], network['biases'])):
        x = x@w+b
        if i < len(network['weights'])-1:
            x = np.tanh(x)
        cache.append(x)
    return x, cache


def fit_frontier(scenarios, config):
    if not scenarios or any(not 1 <= s.scenario_id <= 120 for s in scenarios):
        raise PermissionError('Frontier fit accepts development paths only')
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError('Duplicate frontier training ID')
    real = np.array([[[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns]
                      for o in s.observations] for s in scenarios])
    if real.shape[1:] != (30, 3) or not np.isfinite(real).all():
        raise ValueError('Frontier requires complete finite 30-year paths')
    settings = config['frontier_research']
    targets = np.vstack((grid(step=settings['grid_step']), EQUAL))
    growth, loss = annual_paths(real, targets, config['cost_rate'])
    bundle = dict(train_ids=[s.scenario_id for s in scenarios], settings=settings,
                  state_names=list(STATE_NAMES), gross_paths=np.cumprod(1+real, axis=1).tolist(),
                  annual_returns=real.reshape(-1, 3).tolist(), fits={})
    for trial, setting in enumerate(settings['trials']):
        began = time.perf_counter(); allocations = {}; diagnostics = []
        for h in config['horizons']:
            g, d = growth[h-1], loss[h-1]
            tails = tail_mean(d, axis=1)
            p10 = np.quantile(g, .1, axis=1)
            if setting.get('reference')=='balanced':
                reference=int(np.flatnonzero(np.all(np.isclose(targets,PORTFOLIOS['balanced']),axis=1))[0])
                risk_statistic=np.quantile(d,.9,axis=1)
                budget=max(0,float(risk_statistic[reference])-setting['minimum_p90_reduction'])
            else:
                reference=len(targets)-1; risk_statistic=tails
                budget=tails[-1]*setting['risk_budget_ratio']
            feasible = (risk_statistic <= budget+1e-12) & (p10 >= p10[reference]-settings['maximum_p10_loss'])
            # Explicit infeasibility is retained; never call a relaxed target feasible.
            violation = np.maximum(0, risk_statistic-budget)+np.maximum(0, p10[reference]-settings['maximum_p10_loss']-p10)
            medians = np.median(g, axis=1)
            candidates = np.flatnonzero(feasible) if feasible.any() else np.flatnonzero(violation <= violation.min()+1e-12)
            best = min(candidates, key=lambda i: (-medians[i], tails[i], np.linalg.norm(targets[i]-EQUAL), int(i)))
            allocations[str(h)] = targets[best].tolist()
            diagnostics.append(dict(horizon=h, feasible=bool(feasible[best]), risk_budget=float(budget),
                budget_statistic='p90_drawdown' if setting.get('reference')=='balanced' else 'tail_mean_drawdown',
                training_budget_value=float(risk_statistic[best]),training_p90_drawdown=float(np.quantile(d[best],.9)),
                training_tail_drawdown=float(tails[best]), training_p10_growth=float(p10[best]),
                training_median_growth=float(medians[best]), weights=targets[best].tolist()))
        rng = np.random.default_rng(config['seed'])
        count = settings['training_states']
        hs = rng.choice(config['horizons'], count)
        rem = np.array([rng.integers(1, h) for h in hs])
        strategic = np.array([allocations[str(h)] for h in hs])
        weights = .5*rng.dirichlet(np.ones(3)*2, count)+.5*strategic
        dd = rng.uniform(0, .6, count)
        maximum = dd+rng.uniform(0, 1, count)*(.75-dd)
        choices = np.stack((weights, strategic, np.broadcast_to(PORTFOLIOS['conservative'], weights.shape)), axis=1)
        q, _ = value_estimates(choices, weights, dd, maximum, rem, hs,
                              np.array(bundle['gross_paths']), setting['drawdown_penalty'], config['cost_rate'],
                              settings.get('risk_barrier',0.),settings.get('risk_slack',.01))
        y = q-q[:, :1]
        x = state_features(weights, dd, maximum, rem, hs, config['cost_rate'], strategic)
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
        z = np.clip((x-mean)/scale, -8, 8)
        design = np.column_stack((np.ones(count), z))
        ridge = np.eye(design.shape[1])*settings['ridge']; ridge[0, 0] = 0
        linear = np.linalg.solve(design.T@design+ridge, design.T@y)
        networks = []; trace = []
        output_scale = max(float(y.std()), 1e-4)
        for seed in settings['seeds']:
            net = init_network(np.random.default_rng(seed), settings['hidden_sizes'])
            # The existing network constructor uses the legacy input dimension.
            net['weights'][0] = np.random.default_rng(seed).normal(0, 1/np.sqrt(len(STATE_NAMES)),
                (len(STATE_NAMES), settings['hidden_sizes'][0]))
            opt = Adam(net)
            for epoch in range(settings['epochs']):
                predicted, cache = network_values(net, z)
                residual = predicted-y/output_scale
                opt.update(gradients(net, cache, 2*residual/residual.size), settings['learning_rate'])
            predicted, _ = network_values(net, z)
            trace.append(dict(seed=seed, mse=float(np.mean((predicted*output_scale-y)**2))))
            networks.append(pack_network(net))
        bundle['fits'][str(trial)] = dict(allocations=allocations, diagnostics=diagnostics,
            mean=mean.tolist(), scale=scale.tolist(), linear=linear.tolist(), networks=networks,
            output_scale=output_scale, trace=trace, training_seconds=time.perf_counter()-began)
    return bundle


class FrontierPolicy:
    requires_benefit_gate = True

    def __init__(self, name, model, params, config, horizon):
        if horizon is None or str(horizon) not in model['frontier']['fits']['0']['allocations']:
            raise ValueError('Frontier requires a declared horizon')
        self.name = name; self.horizon = horizon
        self.method=name.replace('frontier_matched_','frontier_')
        bundle = model['frontier']
        if set(bundle['train_ids']) != set(model['train_ids']):
            raise ValueError('Frontier and base training IDs differ')
        self.trial = int(params.get('frontier_trial', 0))
        self.settings = bundle['settings']; self.fit = bundle['fits'][str(self.trial)]
        self.penalty = self.settings['trials'][self.trial]['drawdown_penalty']
        self.initial_target = tuple(self.fit['allocations'][str(horizon)])
        self.buffer = self.settings['buffer']
        self.gross = np.array(bundle['gross_paths'])
        self.annual = np.array(bundle['annual_returns'])
        self.networks = [unpack_network(n) for n in self.fit['networks']]
        self.inference_seconds = 0.; self.decision_calls = 0
        self.requires_benefit_gate = self.method not in ('frontier_annual', 'frontier_hold')

    def decide(self, c):
        began = time.perf_counter()
        try:
            return self._decide(c)
        finally:
            self.inference_seconds += time.perf_counter()-began; self.decision_calls += 1

    def _decide(self, c):
        if self.method == 'frontier_annual':
            return Decision(Action.REBALANCE, ('FRONTIER_ANNUAL',), 'Restore the risk-budget allocation.', self.initial_target)
        wealth = sum(c.holdings)/c.cumulative_inflation
        dd = 1-wealth/c.real_peak
        if not np.isfinite(wealth) or dd >= self.settings['suspension_drawdown']:
            return Decision(Action.HUMAN_REVIEW, ('FRONTIER_SUSPENSION',),
                'Your purchasing-power drawdown exceeds the automated service threshold. A human adviser should reassess your plan. We will hold existing investments while review is pending.')
        if self.method == 'frontier_hold':
            return Decision(Action.KEEP, ('FRONTIER_HOLD',), 'Hold the initial risk-budget allocation.')
        weights = np.array(c.holdings)[None, :]/sum(c.holdings)
        strategic = np.array(self.initial_target)[None, :]
        targets = np.stack((weights, strategic, np.array(PORTFOLIOS['conservative'])[None, :]), axis=1)
        rem = np.array([self.horizon-c.period+1]); hs = np.array([self.horizon])
        fee = c.transaction_cost_rate*.5*np.abs(targets-weights[:, None, :]).sum(axis=2)
        if self.method in ('frontier_mpc', 'frontier_gate'):
            gross = self.gross if self.method == 'frontier_mpc' else (1+self.annual)[:, None, :]
            remaining = rem if self.method == 'frontier_mpc' else np.ones(1, dtype=int)
            q, fee = value_estimates(targets, weights, np.array([dd]), np.array([c.real_max_drawdown]),
                                    remaining, hs, gross, self.penalty, c.transaction_cost_rate,
                                    self.settings.get('risk_barrier',0.),self.settings.get('risk_slack',.01))
        else:
            raw = state_features(weights, np.array([dd]), np.array([c.real_max_drawdown]), rem, hs,
                                 self.settings['training_cost_rate'], strategic)
            z = np.clip((raw-np.array(self.fit['mean']))/np.array(self.fit['scale']), -8, 8)
            if self.method == 'frontier_linear':
                q = np.column_stack((np.ones(1), z))@np.array(self.fit['linear'])
            else:
                q = np.mean([network_values(net, z)[0] for net in self.networks], axis=0)*self.fit['output_scale']
            # Training labels contain the official immediate fee. Correct only that
            # known log-wealth component in fee sensitivity runs; MDD effect remains
            # an approximation, explicitly tested against the exact planner.
            official_fee = self.settings['training_cost_rate']*.5*np.abs(targets-weights[:, None, :]).sum(axis=2)
            q += (np.log1p(-fee)-np.log1p(-official_fee))/hs[:, None]
        if not np.isfinite(q).all():
            return Decision(Action.HUMAN_REVIEW, ('FRONTIER_INVALID_VALUE',),
                'Our allocation value estimates are unreliable. We will hold existing investments while a human adviser reviews your plan.')
        delta = q[0]-q[0, 0]
        # Convert risk-adjusted net value to a pre-cost certainty-equivalent
        # benefit. The engine independently checks the exact immediate fee.
        benefit = 1-(1-fee[0])*np.exp(np.clip(-self.horizon*delta, -100, 100))
        allowed = benefit > fee[0]+self.buffer; allowed[0] = True
        selected = int(np.argmax(np.where(allowed, q[0], -np.inf)))
        codes = ('FRONTIER_'+('PATIENCE', 'STRATEGIC', 'DEFENSIVE')[selected],
                 'LEGAL_ACTIONS_'+str(int(allowed.sum())),
                 'VALUE_ADVANTAGES_'+','.join(f'{x:.6f}' for x in delta))
        if not selected:
            return Decision(Action.KEEP, codes, 'Keep existing holdings: no estimated risk-adjusted improvement clears trading costs and the safety margin.',
                            estimated_benefit=0., estimated_cost=0.)
        target = tuple(targets[0, selected])
        action = Action.REBALANCE if np.allclose(target, c.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        return Decision(action, codes, 'Change the allocation because its estimated purchasing-power growth and drawdown trade-off exceeds trading costs and our safety margin. These estimates use development scenarios; future losses remain possible.',
                        target, float(benefit[selected]), float(fee[0, selected]))
