# RunPod setup

Model work runs on a RunPod GPU pod. Neovim and molten stay on the Mac and connect to a Jupyter server on the pod through an SSH tunnel.

## This project's resources

All pods and the network volume were deleted on 2026-10-26, after checking every file in `data/` on the volume matched the local copy. The table records what was used.

| Item | Value |
|---|---|
| Pod | 7B runs: `ai-collusion-sprint-2` (`bhms915dvcztr5`), 1× RTX PRO 4500 (32 GB), $0.72/hr. 14B replication: `ai-collusion-sprint-14b` (`uxe0a15dpwiehr`), 1× RTX PRO 6000 (96 GB), $2.49/hr, terminated 2026-10-25 after the run. Llama-3.1-8B replication: `ai-collusion-sprint-llama` (`x24qqg97hv9fih`), 1× RTX PRO 4500, $0.72/hr, created 2026-10-25 because the stopped 7B pods' hosts had no free GPU. Both Secure Cloud, EU-RO-1. |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (template `runpod-torch-v280`) |
| CUDA floor | 13.0, because the locked torch uses the CUDA 13 runtime |
| Network volume | `ai-collusion-sprint` (`bs1ne1ykdh`), 80 GB (grown from 50 GB on 2026-10-25 for 14B weights), EU-RO-1, mounted at `/workspace` |
| Ports | `22/tcp` only. Jupyter is reached through the tunnel, not a public proxy URL. |

## Where things live on the pod

| Path | Disk | Survives terminate |
|---|---|---|
| `/workspace/<repo>` (code, `data/`) | network volume | yes |
| `/workspace/hf` (HF cache), `/workspace/.uv-cache` | network volume | yes |
| `/root/venv`, `~/.local/bin/uv` | pod disk | no; `scripts/start_jupyter.sh` rebuilds them in about a minute |

The venv sits on the pod disk because imports from the network volume take minutes; from local disk they take seconds.

## New project or new pod

1. **Mac, once:** `~/.ssh/id_ed25519` is registered on the RunPod account. Neovim's Python host (`~/.virtualenvs/neovim`) has `requests` and `websocket-client`, which remote molten needs.
2. **Create (Claude with the Runpod plugin, or the console):** a network volume in a data center with stock for the GPU, then a pod in that data center with: template `runpod-torch-v280`, a CUDA floor matching the locked torch, `ports: ["22/tcp"]`, `startSsh: true`, `startJupyter: false`, the volume mounted at `/workspace`.
3. **SSH alias:** add or update `Host pod` in `~/.ssh/config` with the pod's `ssh.direct` host and port. The port changes every time the pod restarts.
4. **Sync the repo** (from the repo root on the Mac):
   ```bash
   rsync -rltz --exclude .venv --exclude __pycache__ --exclude data ./ pod:/workspace/<repo>/
   ```
   Use `-rltz`, not `-a`. The network volume refuses `chown`, so `-a` fails with exit code 23.
5. **Start Jupyter:** `ssh pod /workspace/<repo>/scripts/start_jupyter.sh`. It runs without a token, because the server binds to the pod's localhost and only the SSH tunnel reaches it. The project needs `jupyter-server` and `ipykernel` in its dev deps.
6. **Connect:** run `ssh -N -L 8888:localhost:8888 pod` on the Mac, then `:MoltenInit http://localhost:8888` in Neovim. `\mi` starts a local kernel instead.

Rerunning `start_jupyter.sh` restarts the server, which kills any running kernel.

## Syncing code

| You changed | What to do |
|---|---|
| A cell in `src/main.py` | Nothing. Molten sends the cell text to the kernel. |
| A helper in `src/collusion_probes/` | Resync, then restart the kernel, or use `%autoreload 2`. |
| `pyproject.toml` / `uv.lock` | Resync, rerun `start_jupyter.sh`, reconnect molten. |

## Getting results back

`data/` is gitignored and written on the pod. Pull it to the Mac with:

```bash
rsync -rltz pod:/workspace/ai_collusion_sprint/data/ ./data/
```

File names include the model name, so 7B results don't overwrite local 1.5B ones.

## Billing

- **Running pod:** $0.72/hr. Terminating ends every pod charge; this pod has no volume disk of its own.
- **Network volume:** bills hourly per GB from creation until it is deleted, whether or not a pod is attached. Pull `data/` back before deleting it.
