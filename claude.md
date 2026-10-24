@docs/project_scope.md
@docs/prompt_provenance.md

## Editor setup (how code is run)

I run this code interactively, notebook-style, in Neovim with molten-nvim. Locally I use Ghostty and tmux. For model work I SSH into a rented GPU box and work inside tmux there.

- **Cells:** `.py` files are split into cells with `# %% <title>` markers (see `src/main.py`; helpers live in `src/ai_collusion_sprint/`). Keep that structure. Each cell should run on its own, top to bottom, against shared kernel state. Don't wrap logic in `main()` or `if __name__ == "__main__"`, because that hides variables from the kernel.
- **Kernel:** a Jupyter kernel named `ai_collusion_sprint`, which runs this project's uv `.venv`. Add deps with `uv add`. `ipykernel` is in the dev group.
- **Keys:** `\mi` starts the kernel, `\rc` runs the cell under the cursor, `\rn` runs it and moves to the next, and `]x`/`[x` jump between cells.
- **Output:** a cell's output opens in a float when my cursor enters that cell, capped at 20 lines, so keep printed output short. Print summaries and shapes, not whole tensors. A green gutter bar marks cells that have run, and it turns yellow once a cell is edited.
- **Plots:** matplotlib figures show inline as images through the kitty graphics protocol. End plotting cells with `plt.show()`. Keep figures small; they're capped at roughly 100 columns by 12 rows. Plotly won't render.
- **Stale output:** nothing marks output as out of date after a cell is edited. When you change a cell, tell me which later cells I need to re-run.
- **Device:** the same code runs on my Mac and on the GPU box. Pick the device at runtime (`cuda`, then `mps`, then `cpu`) and never hardcode it.
