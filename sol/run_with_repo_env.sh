#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "$repo_root/.env"
set +a

if [[ -n "${HF_WRITE_TOKEN_PERSONAL:-}" ]]; then
  export HF_TOKEN="$HF_WRITE_TOKEN_PERSONAL"
fi

exec "$@"
