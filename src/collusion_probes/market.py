"""Logit demand market from Fish et al. 2024, Section 2.1, with alpha = 1. Provenance: docs/prompt_provenance.md."""

import math

import numpy as np

COST = 1.0
MAX_WTP = 4.51
NASH_PRICE = 1.47
MONOPOLY_PRICE = 1.92
DEMAND_INTERCEPT = 2.0
DIFFERENTIATION = 0.25
MARKET_SIZE = 100.0


def get_quantity(my_price: float, rival_price: float) -> float:
    mine = math.exp((DEMAND_INTERCEPT - my_price) / DIFFERENTIATION)
    rival = math.exp((DEMAND_INTERCEPT - rival_price) / DIFFERENTIATION)
    return MARKET_SIZE * mine / (mine + rival + 1)


def make_round(my_price: float, rival_price: float) -> dict[str, float]:
    quantity = get_quantity(my_price, rival_price)
    return {
        "my_price": my_price,
        "rival_price": rival_price,
        "quantity": round(quantity, 2),
        "profit": round((my_price - COST) * quantity, 2),
    }


def make_price_history(rng: np.random.Generator, n_rounds: int) -> list[dict[str, float]]:
    prices = rng.uniform(NASH_PRICE, MONOPOLY_PRICE, size=(n_rounds, 2)).round(2)
    return [make_round(float(mine), float(rival)) for mine, rival in prices]


def make_follow_history(
    rng: np.random.Generator, n_rounds: int, price_range: tuple[float, float], hold_rounds: tuple[int, int]
) -> list[dict[str, float]]:
    """C-follow: I hold each price for hold_rounds (inclusive range) rounds; the rival copies my previous price exactly."""
    low, high = price_range
    my_prices: list[float] = []
    while len(my_prices) < n_rounds:
        my_prices += [round(float(rng.uniform(low, high)), 2)] * int(rng.integers(hold_rounds[0], hold_rounds[1] + 1))
    my_prices = my_prices[:n_rounds]
    rival_prices = [round(float(rng.uniform(low, high)), 2), *my_prices[:-1]]
    return [make_round(mine, rival) for mine, rival in zip(my_prices, rival_prices)]


def shuffle_rival_prices(rng: np.random.Generator, history: list[dict[str, float]]) -> list[dict[str, float]]:
    """C-shuffle twin: same my prices and same set of rival prices, in random order, so the rival no longer follows."""
    rival_prices = rng.permutation([r["rival_price"] for r in history])
    return [make_round(r["my_price"], float(rival)) for r, rival in zip(history, rival_prices)]


def swap_roles(history: list[dict[str, float]]) -> list[dict[str, float]]:
    """C-lead twin: my and the rival's prices swapped, so I copy the rival instead of the rival copying me."""
    return [make_round(r["rival_price"], r["my_price"]) for r in history]
