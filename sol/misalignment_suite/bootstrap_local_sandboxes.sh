#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state_root="${LOCAL_SUITE_ROOT:-$repo_root/sol/misalignment_suite/.state}"
upstream_root="$state_root/upstreams"
venv_root="$state_root/venvs"
config="$repo_root/sol/misalignment_suite/config.json"

export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
mkdir -p "$upstream_root" "$venv_root"

command -v uv >/dev/null
docker info >/dev/null
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
  [[ "$commit" == "$current" ]]
}

for upstream in instrumental-choices shutdown-avoidance odcv-bench reward-hacking; do
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

main_python="$venv_root/main/bin/python"
if [[ ! -x "$main_python" ]]; then
  uv venv --python 3.12 "$venv_root/main"
fi
uv pip install --python "$main_python" "openai>=2.26" "huggingface_hub>=1.2"
uv pip install --python "$main_python" -e "$upstream_root/reward-hacking/rl-envs"

(
  cd "$upstream_root/instrumental-choices"
  uv sync --locked --python 3.13 --extra dev
)
(
  cd "$upstream_root/shutdown-avoidance"
  uv sync --locked --python 3.13
)

echo "Local CPU-only sandbox environment is ready at $state_root"
