# %% Imports
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from market import make_price_history
from prompt import CUES, build_prompt, parse_price

# %% Config
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # swap to "Qwen/Qwen2.5-7B-Instruct" on the GPU box
DATA_DIR = Path("data")
RUNS_PATH = DATA_DIR / f"runs_{MODEL_ID.split('/')[-1]}.jsonl"
N_PAIRS = 10  # a few hundred on the GPU box
N_ROUNDS = 20  # rounds of price history; Fish et al. use 100
MAX_NEW_TOKENS = 512
SEED = 0

# %% Load model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.eval()
print(f"{MODEL_ID} on {DEVICE}: {model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}")


# %% Completion helper
def complete(prompt: str, do_sample: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample)  # pyright: ignore[reportAttributeAccessIssue]  # transformers stub bug
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


print(complete("Pick a price between 1 and 10. Reply with just the number.", do_sample=False))

# %% Prompt preview
print(build_prompt(make_price_history(np.random.default_rng(SEED), N_ROUNDS), CUES["collusion"])[-600:])


# %% Generate dataset (skipped if RUNS_PATH exists; delete it to regenerate)
@dataclass
class Run:
    pair_id: int
    condition: str
    history: list[dict[str, float]]
    prompt: str
    completion: str
    price: float | None


if RUNS_PATH.exists():
    runs = [Run(**json.loads(line)) for line in RUNS_PATH.read_text().splitlines()]
else:
    rng = np.random.default_rng(SEED)
    _ = torch.manual_seed(SEED)
    runs: list[Run] = []
    for pair_id in range(N_PAIRS):
        history = make_price_history(rng, N_ROUNDS)
        for condition, cue in CUES.items():
            prompt = build_prompt(history, cue)
            completion = complete(prompt)
            runs.append(Run(pair_id, condition, history, prompt, completion, parse_price(completion)))
        print(f"pair {pair_id + 1}/{N_PAIRS}: {[r.price for r in runs[-len(CUES):]]}")
    DATA_DIR.mkdir(exist_ok=True)
    _ = RUNS_PATH.write_text("".join(json.dumps(asdict(run)) + "\n" for run in runs))

for condition in CUES:
    prices = [r.price for r in runs if r.condition == condition and r.price is not None]
    n_runs = sum(r.condition == condition for r in runs)
    print(f"{condition}: {len(prices)}/{n_runs} parsed, mean price {np.mean(prices):.2f}")



    
