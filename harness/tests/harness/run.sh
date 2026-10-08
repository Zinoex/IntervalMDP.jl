#!/bin/sh
# Harness test entry point. Requires python3 >= 3.11; uses julia when present.
# Checks needing the pinned Lean toolchain or a functional CUDA GPU report
# UNAVAILABLE (never PASS). Exit: 0 all passed, 1 failure, 2 PASS-INCOMPLETE (UNAVAILABLE>0).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if ! command -v python3 >/dev/null 2>&1; then
  echo "BLOCKER: python3 not found; harness tests cannot run" >&2
  exit 2
fi
for t in julia lake lean; do
  if command -v "$t" >/dev/null 2>&1; then echo "tool $t: available"; else echo "tool $t: UNAVAILABLE"; fi
done
exec python3 "$HERE/run.py" "$@"
