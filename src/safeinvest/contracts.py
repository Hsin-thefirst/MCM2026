"""Contract v1. See docs/INTERFACES.md for timing, accounting and metric semantics."""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal, Protocol

ASSETS = ("equity", "bond", "bill")
CONTRACT_VERSION = "1.0"
Weights = tuple[float, float, float]
Holdings = tuple[float, float, float]
PortfolioName = Literal["conservative", "balanced", "growth"]
Liquidity = Literal["high", "medium", "low"]


def validate_weights(weights: Weights) -> None:
    if len(weights) != 3 or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("Weights must be three finite nonnegative values")
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Weights must sum to one")


@dataclass(frozen=True)
class MarketObservation:
    scenario_id: int
    period: int
    nominal_returns: tuple[float, float, float]
    inflation: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    observations: tuple[MarketObservation, ...]


@dataclass(frozen=True)
class CustomerProfile:
    profile_id: str
    horizon: int
    loss_tolerance: float
    liquidity: Liquidity


@dataclass(frozen=True)
class PortfolioSpec:
    name: PortfolioName
    weights: Weights

    def __post_init__(self) -> None:
        validate_weights(self.weights)


@dataclass(frozen=True)
class SuitabilityResult:
    profile_id: str
    portfolio: PortfolioName | None  # None means human review
    reason_codes: tuple[str, ...]
    explanation: str


class Action(str, Enum):
    KEEP = "keep"
    REBALANCE = "rebalance"
    CHANGE_TARGET = "change_target"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class PolicyContext:
    period: int
    history: tuple[MarketObservation, ...]  # Exactly periods 1..t-1; never future data
    holdings: Holdings  # Nominal holdings immediately before review
    target_weights: Weights
    cumulative_inflation: float  # Product through t-1; initial value 1
    real_peak: float  # Maximum realized real wealth so far; initial value 100
    review_pending: bool
    transaction_cost_rate: float


@dataclass(frozen=True)
class Decision:
    action: Action
    reason_codes: tuple[str, ...]
    explanation: str
    target_weights: Weights | None = None
    estimated_benefit: float | None = None  # Fraction of pre-trade nominal wealth
    estimated_cost: float | None = None  # Same unit as estimated_benefit
    gate_exempt: bool = False  # Protective de-risking trade; exempt from the benefit inequality

    def __post_init__(self) -> None:
        trades = self.action in (Action.REBALANCE, Action.CHANGE_TARGET)
        if trades != (self.target_weights is not None):
            raise ValueError("Only rebalance/change_target must supply target_weights")
        if self.target_weights is not None:
            validate_weights(self.target_weights)
        for value in (self.estimated_benefit, self.estimated_cost):
            if value is not None and not math.isfinite(value):
                raise ValueError("Estimated values must be finite")
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError("Estimated cost must be nonnegative")


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int = 30
    initial_wealth: float = 100.0
    transaction_cost_rate: float = 0.0025

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= 30:
            raise ValueError("Horizon must be between 1 and 30")
        if not math.isfinite(self.initial_wealth) or self.initial_wealth <= 0:
            raise ValueError("Initial wealth must be positive and finite")
        if not math.isfinite(self.transaction_cost_rate) or not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("Cost rate must be in [0, 1)")


@dataclass(frozen=True)
class TradeResult:
    post_trade_holdings: Holdings
    moved_amount: float
    transaction_cost: float


@dataclass(frozen=True)
class PeriodRecord:
    period: int
    action: Action
    target_weights: Weights
    pre_trade_holdings: Holdings
    post_trade_holdings: Holdings
    end_holdings: Holdings
    moved_amount: float
    transaction_cost: float
    cumulative_inflation: float
    real_wealth: float
    review_pending: bool
    reason_codes: tuple[str, ...]
    explanation: str
    estimated_benefit: float | None = None
    estimated_cost: float | None = None
    gate_exempt: bool = False


@dataclass(frozen=True)
class SimulationResult:
    scenario_id: int
    policy_name: str
    config: SimulationConfig
    initial_target: Weights
    initial_moved_amount: float
    initial_transaction_cost: float
    initial_post_trade_holdings: Holdings
    records: tuple[PeriodRecord, ...]


@dataclass(frozen=True)
class ScenarioMetrics:
    scenario_id: int
    policy_name: str
    horizon: int
    real_cagr: float
    real_max_drawdown: float
    real_return_volatility: float
    terminal_real_wealth: float
    trade_rate: float
    total_nominal_cost: float


@dataclass(frozen=True)
class EvaluationSummary:
    policy_name: str
    scenario_ids: tuple[int, ...]
    horizon: int
    median_real_cagr: float
    p10_real_cagr: float
    median_real_max_drawdown: float
    p90_real_max_drawdown: float
    median_real_return_volatility: float
    mean_trade_rate: float


class Policy(Protocol):
    """Create a fresh policy instance for each scenario; fitted parameters are frozen."""
    name: str

    def decide(self, context: PolicyContext) -> Decision: ...


class SuitabilityRule(Protocol):
    def recommend(self, profile: CustomerProfile) -> SuitabilityResult: ...
