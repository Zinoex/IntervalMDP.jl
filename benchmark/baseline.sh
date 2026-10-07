#!/usr/bin/env bash
# Phase 0 baseline sequence (run from the repository root). Every run is a
# separate Julia process; rounds are interleaved over thread counts so that
# thermal/clock drift is spread over all of them.
#
#   bash benchmark/baseline.sh rounds [TAG...]  # full suite at t = 1,4,8,16 for each round tag
#                                              # (default: "" -rerun1; "" = the baseline itself)
#   bash benchmark/baseline.sh scaling  # strong scaling (bellman!, fixed size), t = 1..16
#   bash benchmark/baseline.sh sizes    # size scaling, t = 1, 6, 16
#   bash benchmark/baseline.sh stream   # machine roofs, t = 1, 6, 16
#   bash benchmark/baseline.sh cuda     # CUDA reference + baseline + re-run
set -u
cd "$(dirname "$0")/.."
# Files are named after the base ref whose src/ and ext/ are measured (Phase 0:
# main = 20fc03b). Refuse to run if src/ or ext/ differ from it.
SHA=${BENCH_BASE_REF:-20fc03b}
if [ -n "$(git diff --stat "$SHA" -- src ext)" ]; then
    echo "src/ or ext/ differ from $SHA; set BENCH_BASE_REF to the commit being measured" >&2
    exit 1
fi
RES=benchmark/results
J="julia --project=benchmark"

case "${1:-rounds}" in
rounds)
    shift
    tags=("$@")
    [ ${#tags[@]} -eq 0 ] && tags=("" "-rerun1")
    for round in "${tags[@]}"; do
        for t in 1 4 8 16; do
            $J --threads=$t benchmark/run.jl --suite full --out $RES/baseline-$SHA-cpu-t$t$round.json
        done
    done
    ;;
scaling)
    for t in 1 2 4 6 8 10 12 14 16; do
        $J --threads=$t benchmark/run.jl --suite scaling --entries bellman --out $RES/scaling-$SHA-cpu-t$t.json
    done
    ;;
sizes)
    for t in 1 6 16; do
        $J --threads=$t benchmark/run.jl --suite sizes --out $RES/sizes-$SHA-cpu-t$t.json
    done
    ;;
stream)
    for t in 1 2 4 6 8 10 12 14 16; do
        $J --threads=$t benchmark/stream.jl --out $RES/stream-t$t.json
    done
    ;;
cuda)
    $J benchmark/run.jl --suite full --backend cuda --eltype all --write-reference --reference-only --out $RES/reference-run-$SHA-cuda.json
    $J benchmark/run.jl --suite full --backend cuda --eltype all --out $RES/baseline-$SHA-cuda.json
    $J benchmark/run.jl --suite full --backend cuda --eltype all --out $RES/baseline-$SHA-cuda-rerun1.json
    ;;
esac
