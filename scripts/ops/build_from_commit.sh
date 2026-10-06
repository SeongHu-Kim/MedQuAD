#!/usr/bin/env bash
# Build an API image from an exact committed tree (not the working tree), labelled with the commit hash.
# Read-only for git: uses `git archive` into a temp dir; no checkout, worktree or index changes.
# Usage: VARIANT=gpu|cpu scripts/ops/build_from_commit.sh <commit>   Owner: service-platform-engineer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT="$(git -C "$ROOT" rev-parse --verify "${1:?usage: build_from_commit.sh <commit>}^{commit}")"
SHORT="${COMMIT:0:12}"
VARIANT="${VARIANT:-gpu}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/medquad-build-$SHORT-XXXX")"
trap 'rm -rf "$TMP"' EXIT
git -C "$ROOT" archive --format=tar "$COMMIT" | tar -x -C "$TMP"
for f in deploy/compose.yaml deploy/docker/api.Dockerfile requirements.lock pyproject.toml; do
  [[ -f "$TMP/$f" ]] || { echo "missing $f in commit $SHORT" >&2; exit 1; }
done
start=$(date +%s)
MEDQUAD_BUILD_ID="$SHORT" docker compose -f "$TMP/deploy/compose.yaml" --profile "$VARIANT" build "api-$VARIANT"
echo "built medquad-api:$VARIANT from $COMMIT in $(( $(date +%s) - start ))s"
docker image inspect -f '{{.Id}} build_id={{index .Config.Labels "medquad.build_id"}}' "medquad-api:$VARIANT"
