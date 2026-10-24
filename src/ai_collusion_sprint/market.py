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


def make_price_history(rng: np.random.Generator, n_rounds: int) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for _ in range(n_rounds):
        my_price, rival_price = (round(float(p), 2) for p in rng.uniform(NASH_PRICE, MONOPOLY_PRICE, size=2))
        quantity = get_quantity(my_price, rival_price)
        history.append({
            "my_price": my_price,
            "rival_price": rival_price,
            "quantity": round(quantity, 2),
            "profit": round((my_price - COST) * quantity, 2),
        })
    return history
