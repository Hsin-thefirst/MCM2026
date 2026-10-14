"""Strict loader for the supplied synthetic data; no model fitting or evaluation."""
import csv
import math
from pathlib import Path
from .contracts import MarketObservation, Scenario

COLUMNS = ("scenario_id", "period", "equity_return", "bond_return", "bill_proxy_return", "inflation")


def load_scenarios(path: str | Path) -> tuple[Scenario, ...]:
    grouped: dict[int, dict[int, MarketObservation]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Expected columns {COLUMNS}")
        for row in reader:
            sid, period = int(row["scenario_id"]), int(row["period"])
            values = tuple(float(row[name]) for name in COLUMNS[2:])
            if not 1 <= sid <= 180 or not 1 <= period <= 30:
                raise ValueError(f"Out-of-range key {(sid, period)}")
            if any(not math.isfinite(v) or v <= -1 for v in values):
                raise ValueError(f"Nonfinite value or value <= -1 at {(sid, period)}")
            periods = grouped.setdefault(sid, {})
            if period in periods:
                raise ValueError(f"Duplicate key {(sid, period)}")
            periods[period] = MarketObservation(sid, period, values[:3], values[3])
    if set(grouped) != set(range(1, 181)):
        raise ValueError("Expected scenarios 1..180")
    for sid, periods in grouped.items():
        if set(periods) != set(range(1, 31)):
            raise ValueError(f"Expected periods 1..30 for scenario {sid}")
    return tuple(Scenario(sid, tuple(grouped[sid][t] for t in range(1, 31))) for sid in range(1, 181))


def split_scenarios(scenarios: tuple[Scenario, ...]) -> tuple[tuple[Scenario, ...], tuple[Scenario, ...]]:
    """Fixed development/final split by scenario ID, never by rows or periods."""
    return (tuple(s for s in scenarios if 1 <= s.scenario_id <= 120),
            tuple(s for s in scenarios if 121 <= s.scenario_id <= 180))
