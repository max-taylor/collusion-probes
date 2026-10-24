"""Run records: one pricing decision each, saved as JSON lines and grouped into pairs by history."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

NO_CUE = "none"  # condition name for the positive-control run on each history


@dataclass
class Run:
    pair_id: int  # runs sharing a pair_id share one price history
    condition: str  # "collusion", "innocent" or NO_CUE
    wording: int | None  # index into CUES[condition]; None for the no-cue run
    history: list[dict[str, float]]
    prompt: str
    completion: str
    price: float | None  # None if no price could be parsed


def load_runs(path: Path) -> list[Run]:
    return [Run(**json.loads(line)) for line in path.read_text().splitlines()]


def save_runs(path: Path, runs: list[Run]) -> None:
    path.parent.mkdir(exist_ok=True)
    _ = path.write_text("".join(json.dumps(asdict(run)) + "\n" for run in runs))


def get_pair_prices(runs: list[Run]) -> dict[int, dict[str, float | None]]:
    """Price per condition for each pair, e.g. {0: {"collusion": 1.81, "innocent": 1.79, "none": 1.62}}."""
    pair_prices: dict[int, dict[str, float | None]] = {}
    for run in runs:
        if run.pair_id not in pair_prices:
            pair_prices[run.pair_id] = {}
        pair_prices[run.pair_id][run.condition] = run.price
    return pair_prices


def keep_pair(price_a: float | None, price_b: float | None, band: tuple[float, float], max_gap: float) -> bool:
    """True if both prices parsed, both sit inside the band, and they are at most max_gap apart."""
    if price_a is None or price_b is None:
        return False
    low, high = band
    both_in_band = low <= price_a <= high and low <= price_b <= high
    return both_in_band and abs(price_a - price_b) <= max_gap
