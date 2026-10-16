"""Small hard-gated policy experts, supervised initialization and REINFORCE.

Markets are exogenous replay paths; all exploration and labels stay inside the
supplied training scenarios. The execution gate is independent of learned scores.
"""
from copy import deepcopy
import time
import numpy as np
from safeinvest.contracts import Action, Decision

EXPERT_NAMES = ('expert_defensive', 'expert_greedy', 'moe_supervised', 'moe_rl', 'moe_rl_static')
LEARNED_NAMES = EXPERT_NAMES[2:]
SEED_NAMES = tuple(f'moe_rl_seed{i}' for i in range(3))
BALANCED = np.array([.50, .35, .15])
DEFENSIVE = np.array([.20, .30, .50])
FEATURE_NAMES = ('equity_weight', 'bond_weight', 'bill_weight', 'drawdown', 'maximum_drawdown',
                 'log_real_wealth', 'remaining_years_scaled', 'horizon_scaled',
                 'equity_mean_5y', 'bond_mean_5y', 'inflation_mean_5y', 'equity_vol_5y')


def enabled(config):
    return config.get('expert_research', {}).get('enabled', False)


def history_features(observations):
    recent = observations[-5:]
    if not recent:
        return np.zeros(4)
    real = np.array([[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns] for o in recent])
    return np.array([real[:, 0].mean(), real[:, 1].mean(),
                     np.mean([o.inflation for o in recent]), real[:, 0].std()])


def features(weights, wealth, peak, remaining, horizon, history, maximum_drawdown=None):
    """Batch features; wealth is relative to the initial purchasing power."""
    maximum_drawdown = 1-wealth/peak if maximum_drawdown is None else maximum_drawdown
    return np.column_stack((weights, 1-wealth/peak, maximum_drawdown, np.log(wealth),
                            remaining/30, horizon/30, history))


def execution_options(weights, drawdown, remaining, model, settings, cost_rate):
    """Return hold/strategic/defensive targets and independently certified actions.

    The benefit is a fractional one-year mean/variance value proxy, not an RL
    value function and not a guarantee. Identical gate for every expert policy.
    """
    n = len(weights)
    targets = np.stack((weights, np.broadcast_to(BALANCED, (n, 3)),
                        np.broadcast_to(DEFENSIVE, (n, 3))), axis=1)
    gamma = (settings['base_gamma'] + settings['drawdown_gamma']*drawdown
             + settings['horizon_gamma']/np.maximum(remaining, 1))
    mu, cov = np.asarray(model['mu']), np.asarray(model['cov'])
    utility = 1 + targets@mu - .5*gamma[:, None]*np.einsum('nai,ij,naj->na', targets, cov, targets)
    fees = cost_rate*.5*np.abs(targets-weights[:, None, :]).sum(axis=2)
    valid = np.isfinite(utility) & (utility > 0)
    benefit = np.divide(utility-utility[:, :1], utility, out=np.zeros_like(utility), where=valid)
    mask = valid & (benefit > fees+settings['buffer'])
    mask[:, 0] = True
    scores = (1-fees)*utility
    return targets, benefit, fees, mask, scores


def init_network(rng, hidden_sizes, static=False):
    dims = [len(FEATURE_NAMES), *hidden_sizes, 3]
    if static:
        return {'static': True, 'weights': [], 'biases': [np.zeros(3)]}
    return {'static': False,
            'weights': [rng.normal(0, np.sqrt(1/a), (a, b)) for a, b in zip(dims[:-1], dims[1:])],
            'biases': [np.zeros(b) for b in dims[1:]]}


def forward(network, x, mask):
    activations = [x]
    if network['static']:
        logits = np.broadcast_to(network['biases'][0], (len(x), 3))
    else:
        z = x
        for i, (w, b) in enumerate(zip(network['weights'], network['biases'])):
            z = z@w+b
            if i < len(network['weights'])-1:
                z = np.tanh(z)
            activations.append(z)
        logits = z
    logits = np.where(mask, logits, -1e30)
    p = np.exp(logits-logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p, activations


def gradients(network, activations, dlogits):
    if network['static']:
        return [], [dlogits.sum(axis=0)]
    dw, db = [], []
    dz = dlogits
    for i in reversed(range(len(network['weights']))):
        dw.append(activations[i].T@dz)
        db.append(dz.sum(axis=0))
        if i:
            dz = (dz@network['weights'][i].T)*(1-activations[i]**2)
    return list(reversed(dw)), list(reversed(db))


class Adam:
    def __init__(self, network):
        self.arrays = network['weights']+network['biases']
        self.m = [np.zeros_like(x) for x in self.arrays]
        self.v = [np.zeros_like(x) for x in self.arrays]
        self.step = 0

    def update(self, grads, lr):
        self.step += 1
        gs = grads[0]+grads[1]
        scale = max(1., np.sqrt(sum(float(np.sum(g*g)) for g in gs))/5)
        for i, (x, g) in enumerate(zip(self.arrays, gs)):
            g = g/scale
            self.m[i] = .9*self.m[i]+.1*g
            self.v[i] = .999*self.v[i]+.001*g*g
            x -= lr*(self.m[i]/(1-.9**self.step))/(np.sqrt(self.v[i]/(1-.999**self.step))+1e-8)
            if not np.isfinite(x).all():
                raise FloatingPointError('Nonfinite expert training parameter')


def pack_network(network):
    return {'static': network['static'], 'weights': [x.tolist() for x in network['weights']],
            'biases': [x.tolist() for x in network['biases']]}


def unpack_network(network):
    return {'static': network['static'], 'weights': [np.array(x) for x in network['weights']],
            'biases': [np.array(x) for x in network['biases']]}


def training_arrays(scenarios):
    if not scenarios or any(not 1 <= s.scenario_id <= 120 for s in scenarios):
        raise PermissionError('Expert fit accepts development scenarios only')
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError('Duplicate training scenario')
    if any(len(s.observations) != 30 for s in scenarios):
        raise ValueError('Expert training requires complete 30-year paths')
    real = np.array([[[(1+r)/(1+o.inflation)-1 for r in o.nominal_returns]
                      for o in s.observations] for s in scenarios])
    history = np.array([[history_features(s.observations[:t]) for t in range(30)] for s in scenarios])
    return real, history


def rollout(real, history, horizons, model, settings, penalty, rng, network=None, scaler=None,
            deterministic=False):
    """Vectorized simulator; independently tested against the scalar ledger engine."""
    n = len(real)
    wealth = np.ones(n)*(1-settings['cost_rate']*(1-BALANCED[2]))
    weights = np.broadcast_to(BALANCED, (n, 3)).copy()
    peak = np.ones(n)
    mdd = 1-wealth
    rewards = np.zeros((30, n))
    samples = []
    for t in range(int(max(horizons))):
        active = np.flatnonzero(t < horizons)
        w, v, pk = weights[active], wealth[active], peak[active]
        h = horizons[active]
        rem = h-t
        targets, benefit, fees, mask, scores = execution_options(w, 1-v/pk, rem, model, settings, settings['cost_rate'])
        if not np.isfinite(scores).all() or np.any(scores <= 0):
            raise ValueError('Invalid utility in expert training; refusing silent fallback')
        x = features(w, v, pk, rem, h, history[active, t], mdd[active])
        if t == 0:
            actions = np.zeros(len(active), dtype=int)
            probs = np.zeros_like(scores); probs[:, 0] = 1
        else:
            if network is None:
                probs = mask/mask.sum(axis=1, keepdims=True)
            else:
                z = np.clip((x-scaler[0])/scaler[1], -8, 8)
                probs, _ = forward(network, z, mask)
            actions = (np.argmax(probs, axis=1) if deterministic else
                       np.minimum((rng.random(len(active))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1), 2))
        # Compute all training-action labels using only this training path's next return.
        after = v[:, None]*(1-fees)
        end = after*(1+np.einsum('nai,ni->na', targets, real[active, t]))
        next_peak = np.maximum(pk[:, None], np.maximum(after, end))
        next_mdd = np.maximum(mdd[active, None], np.maximum(1-after/pk[:, None], 1-end/next_peak))
        immediate = np.log(end/v[:, None])/h[:, None]-penalty*(next_mdd-mdd[active, None])
        if t:
            label = np.argmax(np.where(mask, immediate, -np.inf), axis=1)
            samples.append((t, active.copy(), x, mask, actions, label))
        index = np.arange(len(active))
        selected = targets[index, actions]
        weights[active] = selected*(1+real[active, t])/(1+np.sum(selected*real[active, t], axis=1))[:, None]
        wealth[active] = end[index, actions]
        peak[active] = next_peak[index, actions]
        rewards[t, active] = immediate[index, actions]
        mdd[active] = next_mdd[index, actions]
    # Include the common initial fee in episode-level training diagnostics.
    objective = np.log(wealth)/horizons-penalty*mdd
    return samples, rewards, objective, wealth, mdd


def fit_expert_bundle(scenarios, model, config, penalties=None):
    settings = dict(config['expert_research'], cost_rate=config['cost_rate'])
    if len(settings['seeds']) != 3 or len(set(settings['seeds'])) != 3:
        raise ValueError('Expert protocol requires three distinct predeclared seeds')
    real, history = training_arrays(scenarios)
    horizons = np.asarray(config['horizons'], dtype=int)
    if np.any((horizons < 2) | (horizons > 30)):
        raise ValueError('Expert training horizons must be between 2 and 30')
    bundle = {'train_ids': [s.scenario_id for s in scenarios], 'feature_names': list(FEATURE_NAMES),
              'settings': settings, 'fits': {}}
    for penalty in (settings['drawdown_penalties'] if penalties is None else sorted(set(penalties))):
        began = time.perf_counter()
        # A fixed exploratory behavior provides a training-only standardizer and labels.
        rng = np.random.default_rng(config['seed'])
        data = []
        for h in horizons:
            samples, *_ = rollout(real, history, np.full(len(real), h), model, settings, penalty, rng)
            data.extend(samples)
        x = np.concatenate([s[2] for s in data])
        mask = np.concatenate([s[3] for s in data])
        labels = np.concatenate([s[5] for s in data])
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
        z = np.clip((x-mean)/scale, -8, 8)
        fits = {name: [] for name in LEARNED_NAMES}
        traces = []
        for seed in settings['seeds']:
            rng = np.random.default_rng(seed)
            supervised = init_network(rng, settings['hidden_sizes'])
            opt = Adam(supervised)
            for epoch in range(settings['supervised_epochs']):
                p, cache = forward(supervised, z, mask)
                dz = p.copy(); dz[np.arange(len(z)), labels] -= 1
                opt.update(gradients(supervised, cache, dz/len(z)), settings['learning_rate'])
            fits['moe_supervised'].append(pack_network(supervised))
            for name in ('moe_rl', 'moe_rl_static'):
                if name == 'moe_rl':
                    network = deepcopy(supervised)
                else:
                    network = init_network(rng, [], static=True)
                    warm = Adam(network)
                    for _ in range(settings['supervised_epochs']):
                        p, cache = forward(network, z, mask)
                        dz = p.copy(); dz[np.arange(len(z)), labels] -= 1
                        warm.update(gradients(network, cache, dz/len(z)), settings['learning_rate'])
                opt = Adam(network)
                # Independent fixed streams per method; no seed is selected by validation.
                train_rng = np.random.default_rng(seed+10000)
                for epoch in range(settings['rl_epochs']):
                    hs = train_rng.choice(horizons, len(real))
                    samples, rewards, objective, _, _ = rollout(real, history, hs, model, settings,
                                                               penalty, train_rng, network, (mean, scale))
                    returns = np.cumsum(rewards[::-1], axis=0)[::-1]
                    xs, ms, acts, advs = [], [], [], []
                    for t, active, raw, allowed, actions, _ in samples:
                        ret = returns[t, active]
                        # Leave-one-out baselines within horizon/time; excludes own action.
                        advantage = ret.copy()
                        for h in np.unique(hs[active]):
                            group = hs[active] == h
                            size = int(group.sum())
                            if size > 1:
                                advantage[group] -= (ret[group].sum()-ret[group])/(size-1)
                        xs.append(raw); ms.append(allowed); acts.append(actions); advs.append(advantage)
                    raw = np.concatenate(xs); allowed = np.concatenate(ms)
                    actions = np.concatenate(acts); advantage = np.concatenate(advs)
                    zz = np.clip((raw-mean)/scale, -8, 8)
                    p, cache = forward(network, zz, allowed)
                    # Scale only, without global centering, to preserve the horizon baseline.
                    advantage /= max(float(advantage.std()), 1e-4)
                    dz = p.copy(); dz[np.arange(len(p)), actions] -= 1
                    dz *= advantage[:, None]
                    logp = np.log(np.maximum(p, 1e-30))
                    dz += settings['entropy_bonus']*p*(logp-np.sum(p*logp, axis=1, keepdims=True))
                    opt.update(gradients(network, cache, dz/len(p)), settings['learning_rate'])
                    if epoch == 0 or (epoch+1) % 20 == 0:
                        traces.append(dict(policy=name, seed=seed, epoch=epoch+1,
                                           mean_objective=float(objective.mean()),
                                           mean_entropy=float(-np.sum(p*logp, axis=1).mean())))
                fits[name].append(pack_network(network))
        bundle['fits'][str(penalty)] = dict(mean=mean.tolist(), scale=scale.tolist(), networks=fits,
            seeds=settings['seeds'], trace=traces, training_seconds=time.perf_counter()-began,
            training_failures=0, training_state_count=len(x), drawdown_penalty=penalty,
            label_counts=np.bincount(labels,minlength=3).tolist(),
            legal_action_counts=np.bincount(mask.sum(axis=1),minlength=4).tolist())
    return bundle


class ExpertPolicy:
    requires_benefit_gate = True

    def __init__(self, name, model, params, config, horizon):
        if horizon is None or not 1 <= horizon <= 30:
            raise ValueError('Expert policies require an explicit horizon')
        self.name, self.model, self.horizon = name, model, horizon
        self.settings = config['expert_research']
        self.initial_wealth = config['initial_wealth']
        self.initial_target = tuple(BALANCED)
        self.buffer = self.settings['buffer']
        self.networks = []
        self.inference_seconds = 0.
        self.decision_calls = 0
        if name in LEARNED_NAMES or name in SEED_NAMES:
            bundle = model['experts']
            if set(bundle['train_ids']) != set(model['train_ids']):
                raise ValueError('Expert and base training IDs differ')
            fit = bundle['fits'][str(params['drawdown_penalty'])]
            base_name = 'moe_rl' if name in SEED_NAMES else name
            nets = fit['networks'][base_name]
            if name in SEED_NAMES:
                nets = [nets[int(name[-1])]]
            self.networks = [unpack_network(net) for net in nets]
            self.mean, self.scale = np.array(fit['mean']), np.array(fit['scale'])

    def decide(self, c):
        began = time.perf_counter()
        try:
            return self._decide(c)
        finally:
            self.inference_seconds += time.perf_counter()-began
            self.decision_calls += 1

    def _decide(self, c):
        v = sum(c.holdings)
        weights = np.array(c.holdings)[None, :]/v
        wealth = np.array([v/c.cumulative_inflation])
        peak = np.array([c.real_peak])
        remaining = np.array([self.horizon-c.period+1])
        targets, benefit, fees, mask, scores = execution_options(weights, 1-wealth/peak, remaining,
            self.model, self.settings, c.transaction_cost_rate)
        if not np.isfinite(scores).all() or np.any(scores <= 0):
            return Decision(Action.HUMAN_REVIEW, ('EXPERT_INVALID_ESTIMATE',),
                            'Our allocation estimates are unreliable. A human adviser should review your plan. We will hold existing investments while review is pending.')
        if self.name == 'expert_defensive':
            selected = 2 if mask[0, 2] else 0
            probabilities = np.eye(3)[selected]
        elif self.name == 'expert_greedy':
            selected = int(np.argmax(np.where(mask[0], scores[0], -np.inf)))
            probabilities = np.eye(3)[selected]
        else:
            x = features(weights, wealth/self.initial_wealth, peak/self.initial_wealth, remaining,
                         np.array([self.horizon]), history_features(c.history)[None, :],
                         np.array([c.real_max_drawdown]))
            z = np.clip((x-self.mean)/self.scale, -8, 8)
            probabilities = np.mean([forward(net, z, mask)[0][0] for net in self.networks], axis=0)
            if not np.isfinite(probabilities).all():
                return Decision(Action.HUMAN_REVIEW, ('EXPERT_INVALID_NETWORK',),
                                'The automated model cannot provide a reliable decision. We will hold your investments while a human adviser reviews your plan.')
            selected = int(np.argmax(probabilities))
        reasons = ('EXPERT_'+('PATIENCE', 'STRATEGIC', 'DEFENSIVE')[selected],
                   'LEGAL_ACTIONS_'+str(int(mask.sum())),
                   'GATE_PROBABILITIES_'+','.join(f'{p:.4f}' for p in probabilities))
        if selected == 0:
            return Decision(Action.KEEP, reasons+('NO_CERTIFIED_SWITCH',),
                            'We will keep your current investments. Our expert policy does not select an allocation change that justifies trading costs and the safety margin.',
                            estimated_benefit=0., estimated_cost=0.)
        target = tuple(targets[0, selected])
        action = Action.REBALANCE if np.allclose(target, c.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        explanation = ('We propose a more defensive allocation because its estimated risk-adjusted benefit exceeds trading costs and our safety margin. Your remaining horizon and observed losses affect this assessment. Losses remain possible.'
                       if selected == 2 else
                       'We propose restoring the strategic allocation. Its estimated risk-adjusted benefit exceeds trading costs and our safety margin, given your remaining horizon and observed losses. Future returns remain uncertain.')
        return Decision(action, reasons+('BENEFIT_EXCEEDS_COST',), explanation, target,
                        float(benefit[0, selected]), float(fees[0, selected]))
