#!/bin/bash
# Run on the pod. Builds the venv on local disk (imports from the network volume are slow)
# and starts a Jupyter server in tmux session "jupyter" for molten to connect to.
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export UV_CACHE_DIR=/workspace/.uv-cache UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/workspace/hf
UV=~/.local/bin/uv

[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
cd "$REPO_DIR"
$UV sync -q
$UV run python -m ipykernel install --user --name "$(basename "$REPO_DIR")" >/dev/null

# No token: the server binds to the pod's localhost, so only the SSH tunnel reaches it
tmux kill-session -t jupyter 2>/dev/null || true
tmux new -d -s jupyter "cd $REPO_DIR && UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT HF_HOME=$HF_HOME \
  $UV run jupyter server --allow-root --no-browser --ip 127.0.0.1 --port 8888 \
  --IdentityProvider.token='' 2>&1 | tee /workspace/jupyter.log"
echo "connect: ssh -N -L 8888:localhost:8888 pod, then :MoltenInit http://localhost:8888"
