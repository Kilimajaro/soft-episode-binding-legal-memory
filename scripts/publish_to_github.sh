#!/usr/bin/env bash
# Push current repo state to GitHub (run from repository root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash "$ROOT/scripts/build_publication_packages.sh"
git add -A
if git diff --cached --quiet; then
  echo "[publish] no changes to commit"
else
  git commit -m "Sync publication packages and IPM revision artifacts"
fi
git push -u origin "$(git branch --show-current)"
echo "[publish] pushed $(git branch --show-current)"
