#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_root="${SUITE_ROOT:-/workspace/misalignment-suite}"
upstream_root="$suite_root/upstreams"
venv_root="$suite_root/venvs"
config="$repo_root/sol/misalignment_suite/config.json"

export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
export UV_LINK_MODE=copy
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "$upstream_root" "$venv_root" "$HF_HOME"

if ! command -v unzip >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y unzip git curl ca-certificates
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v bun >/dev/null 2>&1; then
  curl -fsSL https://bun.sh/install | bash
fi

uv python install 3.12 3.13

clone_pinned() {
  local name="$1"
  local url commit destination current
  url="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstreams"][sys.argv[2]]["url"])' "$config" "$name")"
  commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstreams"][sys.argv[2]]["commit"])' "$config" "$name")"
  destination="$upstream_root/$name"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach FETCH_HEAD
  current="$(git -C "$destination" rev-parse HEAD)"
  [[ "$commit" == "$current" ]] || {
    echo "Pinned commit mismatch for $name: wanted $commit, got $current" >&2
    exit 1
  }
}

for upstream in inspect-evals inspect-petri instrumental-choices mask-reference odcv-bench reward-hacking shutdown-avoidance toolalignbench; do
  clone_pinned "$upstream"
done

for reward_patch in \
  "$repo_root/sol/misalignment_suite/patches/reward-hacking-chat-api.patch" \
  "$repo_root/sol/misalignment_suite/patches/reward-hacking-bounds.patch"; do
  if git -C "$upstream_root/reward-hacking" apply --check "$reward_patch"; then
    git -C "$upstream_root/reward-hacking" apply "$reward_patch"
  elif ! git -C "$upstream_root/reward-hacking" apply --reverse --check "$reward_patch"; then
    echo "Reward-hacking patch is neither applicable nor already applied: $reward_patch" >&2
    exit 1
  fi
done

toolalign_patch="$repo_root/sol/misalignment_suite/patches/toolalignbench-local-bounds.patch"
if git -C "$upstream_root/toolalignbench" apply --check "$toolalign_patch"; then
  git -C "$upstream_root/toolalignbench" apply "$toolalign_patch"
elif ! git -C "$upstream_root/toolalignbench" apply --reverse --check "$toolalign_patch"; then
  echo "ToolAlignBench patch is neither applicable nor already applied: $toolalign_patch" >&2
  exit 1
fi

main_python="$venv_root/main/bin/python"
if [[ ! -x "$main_python" ]]; then
  uv venv --python 3.12 "$venv_root/main"
fi
uv pip install --python "$main_python" \
  "vllm==0.24.0" \
  "huggingface_hub>=1.2.0" \
  "safetensors>=0.5" \
  "anthropic>=0.75" \
  "openai>=2.26" \
  "beautifulsoup4>=4.13"
uv pip install --python "$main_python" \
  -e "$upstream_root/inspect-evals[agentic_misalignment]" \
  -e "$upstream_root/inspect-petri" \
  -e "$upstream_root/reward-hacking/rl-envs"

(
  cd "$upstream_root/instrumental-choices"
  uv sync --locked --python 3.13 --extra dev
)
(
  cd "$upstream_root/shutdown-avoidance"
  uv sync --locked --python 3.13
)
(
  cd "$upstream_root/toolalignbench/runner"
  bun install --frozen-lockfile
)

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y docker.io docker-compose-plugin || apt-get install -y docker.io docker-compose
fi

docker_runtime="unavailable"
if ! docker info >/dev/null 2>&1; then
  mkdir -p /var/lib/docker /var/run
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --storage-driver=vfs \
    --iptables=false \
    --ip6tables=false \
    --ip-masq=false \
    --bridge=none \
    >"$suite_root/dockerd.log" 2>&1 &
  for _ in $(seq 1 30); do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done
fi
if docker info >/dev/null 2>&1; then
  docker_runtime="daemon-only"
fi
printf '%s\n' "$docker_runtime" >"$suite_root/docker_runtime.txt"
if [[ "$docker_runtime" != "available" ]]; then
  echo "WARN: this RunPod container cannot start nested benchmark containers; use the hybrid local-sandbox runner."
fi

if [[ -z "${HF_TOKEN:-}" && -n "${HF_WRITE_TOKEN_PERSONAL:-}" ]]; then
  export HF_TOKEN="$HF_WRITE_TOKEN_PERSONAL"
fi
"$main_python" "$repo_root/sol/misalignment_suite/prepare_adapters.py" \
  --output-root "$repo_root/sol/loras" \
  --token "${HF_TOKEN:-}"

"$main_python" - <<'PY'
import json
import platform
import torch
import vllm
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY

echo "Remote evaluation environment is ready at $suite_root"
