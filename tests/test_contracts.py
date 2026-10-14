import unittest
from safeinvest.contracts import Action, Decision, PortfolioSpec, SimulationConfig, validate_weights


class ContractTests(unittest.TestCase):
    def test_long_only_fully_invested_weights(self):
        validate_weights((0.2, 0.3, 0.5))
        for weights in ((-0.1, 0.5, 0.6), (0.2, 0.3, 0.4), (float('nan'), 0, 1), (0.5, 0.5)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_weights(weights)
        with self.assertRaises(ValueError):
            PortfolioSpec('balanced', (0.5, 0.4, 0.2))

    def test_decision_target_matches_action(self):
        Decision(Action.KEEP, ('BENEFIT_BELOW_COST',), 'Keep current holdings.')
        Decision(Action.REBALANCE, ('REBALANCE_BENEFICIAL',), 'Restore target.', (0.5, 0.35, 0.15))
        for action, target in ((Action.KEEP, (1, 0, 0)), (Action.HUMAN_REVIEW, (1, 0, 0)),
                               (Action.REBALANCE, None), (Action.CHANGE_TARGET, None)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                Decision(action, (), 'Invalid.', target)

    def test_finite_estimates_and_nonnegative_cost(self):
        for kwargs in ({'estimated_cost': -0.01}, {'estimated_cost': float('inf')},
                       {'estimated_benefit': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Decision(Action.KEEP, (), 'Invalid.', **kwargs)

    def test_simulation_config_bounds(self):
        for kwargs in ({'horizon': 0}, {'horizon': 31}, {'initial_wealth': 0},
                       {'initial_wealth': float('nan')}, {'transaction_cost_rate': -0.1},
                       {'transaction_cost_rate': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimulationConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
