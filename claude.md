@docs/project_scope.md
@docs/prompt_provenance.md
@docs/runpod_setup.md

## Editor setup (how code is run)

I run this code interactively, notebook-style, in Neovim with molten-nvim. Locally I use Ghostty and tmux. For model work, local molten connects to a Jupyter server on a rented RunPod GPU box through an SSH tunnel (`:MoltenInit http://localhost:8888`, no token). Neovim and my dotfiles stay local. The kernel runs on the pod, so imports from `src/` and files written to `data/` live on the pod; sync `src/` there after editing helpers.

- **Cells:** `.py` files are split into cells with `# %% <title>` markers (see `src/main.py`; helpers live in `src/collusion_probes/`). Keep that structure. Each cell should run on its own, top to bottom, against shared kernel state. Don't wrap logic in `main()` or `if __name__ == "__main__"`, because that hides variables from the kernel.
- **Kernel:** a Jupyter kernel named after the repo folder (`collusion-probes`), which runs this project's uv `.venv`. Add deps with `uv add`. `ipykernel` is in the dev group.
- **Keys:** `\mi` starts the kernel, `\rc` runs the cell under the cursor, `\rn` runs it and moves to the next, and `]x`/`[x` jump between cells.
- **Output:** a cell's output opens in a float when my cursor enters that cell, capped at 20 lines, so keep printed output short. Print summaries and shapes, not whole tensors. A green gutter bar marks cells that have run, and it turns yellow once a cell is edited.
- **Plots:** matplotlib figures show inline as images through the kitty graphics protocol. End plotting cells with `plt.show()`. Keep figures small; they're capped at roughly 100 columns by 12 rows. Plotly won't render.
- **Stale output:** nothing marks output as out of date after a cell is edited. When you change a cell, tell me which later cells I need to re-run.
- **Device:** the same code runs on my Mac and on the GPU box. Pick the device at runtime (`cuda`, then `mps`, then `cpu`) and never hardcode it.

## Experiment changes

Any change to the experimental design (cues, prompt, price band, pair filter, split, thresholds, model) gets a dated entry under `## Change log` at the bottom of `DECISIONS.md`: what changed, why, and the data behind it. Never edit a logged value in its table; add a change-log line instead. Rename superseded data on the pod (e.g. `data/pilot1_*`) rather than deleting it.

## Findings

Results and their meaning go in `FINDINGS.md` as they come in: one dated entry per result, with the numbers, what they mean and the chart path. Don't leave interpretation only in chat.
