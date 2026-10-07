## Machine roofs (measured, benchmark/stream.jl)

| threads | DRAM read GB/s | DRAM copy GB/s | DRAM triad GB/s | L2 read GB/s | in-L1 FMA GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.5 | 15.9 | 17.6 | 60.5 | 12.9 |
| 2 | 11.6 | 13.8 | 12.2 | 12.8 | 14.5 |
| 4 | 20.1 | 24.0 | 24.3 | 22.6 | 24.9 |
| 6 | 26.2 | 31.4 | 38.2 | 22.3 | 55.8 |
| 8 | 43.4 | 37.1 | 40.3 | 36.1 | 66.8 |
| 10 | 28.4 | 27.9 | 25.2 | 48.0 | 124.2 |
| 12 | 28.7 | 21.6 | 37.8 | 52.4 | 160.2 |
| 14 | 48.1 | 36.3 | 30.8 | 63.5 | 153.9 |
| 16 | 43.5 | 32.5 | 33.5 | 58.7 | 93.4 |

## Strong scaling (fixed size, `bellman!` steady state)

| case | entry | t=1 | t=2 | t=4 | t=6 | t=8 | t=10 | t=12 | t=14 | t=16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imdp-dense-n4000-a1 | bellman | 77.97 ms (×1.00, eff 100%) | 44.14 ms (×1.77, eff 88%) | 25.56 ms (×3.05, eff 76%) | 17.11 ms (×4.56, eff 76%) | 17.30 ms (×4.51, eff 56%) | 12.67 ms (×6.15, eff 62%) | 8.29 ms (×9.41, eff 78%) | 7.75 ms (×10.05, eff 72%) | 9.75 ms (×7.99, eff 50%) |
| imdp-dense-n4000-a4 | bellman | 312.66 ms (×1.00, eff 100%) | 167.76 ms (×1.86, eff 93%) | 92.58 ms (×3.38, eff 84%) | 68.78 ms (×4.55, eff 76%) | 54.50 ms (×5.74, eff 72%) | 50.10 ms (×6.24, eff 62%) | 43.83 ms (×7.13, eff 59%) | 40.96 ms (×7.63, eff 55%) | 41.95 ms (×7.45, eff 47%) |
| imdp-sparse-n100000-nnz10-a1 | bellman | 63.21 ms (×1.00, eff 100%) | 35.35 ms (×1.79, eff 89%) | 18.81 ms (×3.36, eff 84%) | 11.64 ms (×5.43, eff 90%) | 12.80 ms (×4.94, eff 62%) | 9.68 ms (×6.53, eff 65%) | 4.20 ms (×15.05, eff 125%) | 3.60 ms (×17.57, eff 125%) | 9.78 ms (×6.46, eff 40%) |
| imdp-sparse-n100000-nnz100-a1 | bellman | 431.63 ms (×1.00, eff 100%) | 219.25 ms (×1.97, eff 98%) | 112.30 ms (×3.84, eff 96%) | 75.66 ms (×5.70, eff 95%) | 82.94 ms (×5.20, eff 65%) | 66.07 ms (×6.53, eff 65%) | 55.72 ms (×7.75, eff 65%) | 47.66 ms (×9.06, eff 65%) | 46.49 ms (×9.28, eff 58%) |

Cell = median (speed-up vs 1 thread, parallel efficiency = speed-up / threads). Threads are pinned compactly: t ≤ 6 P-cores only; t = 8 adds 2 E-cores; t = 14 all P+E; t = 16 adds the 2 LP-E cores.

## Size scaling (`bellman!`, 1 action)

| case | threads | n | nnz | median | ns per nnz | achieved GB/s (model) | GFLOP/s (model) |
|---|---:|---:|---:|---:|---:|---:|---:|
| size-imdp-dense-n250-a1 | 1 | 250 | 62500 | 206.5 µs | 3.30 | 4.8 | 1.21 |
| size-imdp-dense-n500-a1 | 1 | 500 | 250000 | 827.0 µs | 3.31 | 4.8 | 1.21 |
| size-imdp-dense-n1000-a1 | 1 | 1000 | 1000000 | 3.47 ms | 3.47 | 4.6 | 1.15 |
| size-imdp-dense-n2000-a1 | 1 | 2000 | 4000000 | 18.57 ms | 4.64 | 3.4 | 0.86 |
| size-imdp-dense-n4000-a1 | 1 | 4000 | 16000000 | 77.10 ms | 4.82 | 3.3 | 0.83 |
| size-imdp-dense-n8000-a1 | 1 | 8000 | 64000000 | 313.23 ms | 4.89 | 3.3 | 0.82 |
| size-imdp-sparse-n1000-nnz10-a1 | 1 | 1000 | 10000 | 541.8 µs | 54.18 | 0.4 | 0.07 |
| size-imdp-sparse-n10000-nnz10-a1 | 1 | 10000 | 100000 | 6.22 ms | 62.16 | 0.3 | 0.06 |
| size-imdp-sparse-n100000-nnz10-a1 | 1 | 100000 | 1000000 | 62.73 ms | 62.73 | 0.3 | 0.06 |
| size-imdp-sparse-n1000000-nnz10-a1 | 1 | 1000000 | 10000000 | 650.22 ms | 65.02 | 0.3 | 0.06 |
| size-imdp-sparse-n1000-nnz100-a1 | 1 | 1000 | 100000 | 4.05 ms | 40.52 | 0.5 | 0.10 |
| size-imdp-sparse-n10000-nnz100-a1 | 1 | 10000 | 1000000 | 44.85 ms | 44.85 | 0.4 | 0.09 |
| size-imdp-sparse-n100000-nnz100-a1 | 1 | 100000 | 10000000 | 467.57 ms | 46.76 | 0.4 | 0.09 |
| size-imdp-dense-n250-a1 | 6 | 250 | 62500 | 37.2 µs | 0.60 | 26.9 | 6.71 |
| size-imdp-dense-n500-a1 | 6 | 500 | 250000 | 135.1 µs | 0.54 | 29.6 | 7.40 |
| size-imdp-dense-n1000-a1 | 6 | 1000 | 1000000 | 1.19 ms | 1.19 | 13.4 | 3.36 |
| size-imdp-dense-n2000-a1 | 6 | 2000 | 4000000 | 4.12 ms | 1.03 | 15.5 | 3.89 |
| size-imdp-dense-n4000-a1 | 6 | 4000 | 16000000 | 17.22 ms | 1.08 | 14.9 | 3.72 |
| size-imdp-dense-n8000-a1 | 6 | 8000 | 64000000 | 71.75 ms | 1.12 | 14.3 | 3.57 |
| size-imdp-sparse-n1000-nnz10-a1 | 6 | 1000 | 10000 | 160.5 µs | 16.05 | 1.3 | 0.25 |
| size-imdp-sparse-n10000-nnz10-a1 | 6 | 10000 | 100000 | 1.09 ms | 10.93 | 1.9 | 0.37 |
| size-imdp-sparse-n100000-nnz10-a1 | 6 | 100000 | 1000000 | 11.46 ms | 11.46 | 1.8 | 0.35 |
| size-imdp-sparse-n1000000-nnz10-a1 | 6 | 1000000 | 10000000 | 118.38 ms | 11.84 | 1.7 | 0.34 |
| size-imdp-sparse-n1000-nnz100-a1 | 6 | 1000 | 100000 | 1.00 ms | 10.03 | 2.0 | 0.40 |
| size-imdp-sparse-n10000-nnz100-a1 | 6 | 10000 | 1000000 | 7.46 ms | 7.46 | 2.7 | 0.54 |
| size-imdp-sparse-n100000-nnz100-a1 | 6 | 100000 | 10000000 | 74.69 ms | 7.47 | 2.7 | 0.54 |
| size-imdp-dense-n250-a1 | 16 | 250 | 62500 | 43.7 µs | 0.70 | 22.9 | 5.73 |
| size-imdp-dense-n500-a1 | 16 | 500 | 250000 | 186.5 µs | 0.75 | 21.5 | 5.36 |
| size-imdp-dense-n1000-a1 | 16 | 1000 | 1000000 | 770.4 µs | 0.77 | 20.8 | 5.19 |
| size-imdp-dense-n2000-a1 | 16 | 2000 | 4000000 | 2.98 ms | 0.75 | 21.5 | 5.37 |
| size-imdp-dense-n4000-a1 | 16 | 4000 | 16000000 | 9.62 ms | 0.60 | 26.6 | 6.66 |
| size-imdp-dense-n8000-a1 | 16 | 8000 | 64000000 | 41.62 ms | 0.65 | 24.6 | 6.15 |
| size-imdp-sparse-n1000-nnz10-a1 | 16 | 1000 | 10000 | 93.8 µs | 9.38 | 2.2 | 0.43 |
| size-imdp-sparse-n10000-nnz10-a1 | 16 | 10000 | 100000 | 786.3 µs | 7.86 | 2.6 | 0.51 |
| size-imdp-sparse-n100000-nnz10-a1 | 16 | 100000 | 1000000 | 7.72 ms | 7.72 | 2.6 | 0.52 |
| size-imdp-sparse-n1000000-nnz10-a1 | 16 | 1000000 | 10000000 | 189.81 ms | 18.98 | 1.1 | 0.21 |
| size-imdp-sparse-n1000-nnz100-a1 | 16 | 1000 | 100000 | 613.8 µs | 6.14 | 3.3 | 0.65 |
| size-imdp-sparse-n10000-nnz100-a1 | 16 | 10000 | 1000000 | 4.24 ms | 4.24 | 4.7 | 0.94 |
| size-imdp-sparse-n100000-nnz100-a1 | 16 | 100000 | 10000000 | 49.11 ms | 4.91 | 4.1 | 0.81 |

## Roofline estimate (`bellman!` entries of the baseline)

| case | threads | median | data MiB | GB/s (model) | % of roof | GFLOP/s (model) | AI flop/B | allocs/call | class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| imdp-dense-n100-a1 | 1 | 35.0 µs | 0.2 | 4.6 | 8% (L2) | 1.14 | 0.25 | 0 | overhead-bound |
| imdp-dense-n100-a4 | 1 | 138.1 µs | 0.6 | 4.6 | 8% (L2) | 1.16 | 0.25 | 0 | compute/latency-bound |
| imdp-dense-n1000-a1 | 1 | 3.51 ms | 15.3 | 4.6 | 8% (L2) | 1.14 | 0.25 | 0 | compute/latency-bound |
| imdp-dense-n1000-a4 | 1 | 18.48 ms | 61.0 | 3.5 | 28% (DRAM) | 0.87 | 0.25 | 0 | compute/latency-bound |
| imdp-dense-n4000-a1 | 1 | 78.04 ms | 244.1 | 3.3 | 26% (DRAM) | 0.82 | 0.25 | 0 | compute/latency-bound |
| imdp-dense-n4000-a4 | 1 | 312.53 ms | 976.6 | 3.3 | 26% (DRAM) | 0.82 | 0.25 | 0 | compute/latency-bound |
| imdp-sparse-n10000-nnz10-a1 | 1 | 6.31 ms | 1.9 | 0.3 | 1% (L2) | 0.06 | 0.20 | 20000 | compute/latency-bound |
| imdp-sparse-n10000-nnz10-a4 | 1 | 25.22 ms | 7.8 | 0.3 | 1% (L2) | 0.06 | 0.20 | 80000 | compute/latency-bound |
| imdp-sparse-n10000-nnz100-a1 | 1 | 41.09 ms | 19.1 | 0.5 | 1% (L2) | 0.10 | 0.20 | 20000 | compute/latency-bound |
| imdp-sparse-n10000-nnz100-a4 | 1 | 165.76 ms | 76.4 | 0.5 | 4% (DRAM) | 0.10 | 0.20 | 80000 | compute/latency-bound |
| imdp-sparse-n100000-nnz10-a1 | 1 | 64.79 ms | 19.5 | 0.3 | 1% (L2) | 0.06 | 0.20 | 200000 | compute/latency-bound |
| imdp-sparse-n100000-nnz10-a4 | 1 | 267.02 ms | 77.8 | 0.3 | 2% (DRAM) | 0.06 | 0.20 | 800000 | compute/latency-bound |
| imdp-sparse-n100000-nnz100-a1 | 1 | 432.05 ms | 191.1 | 0.5 | 4% (DRAM) | 0.09 | 0.20 | 200000 | compute/latency-bound |
| imdp-sparse-n100000-nnz100-a4 | 1 | 1.74 s | 764.5 | 0.5 | 4% (DRAM) | 0.09 | 0.20 | 800000 | compute/latency-bound |
| product-imdp-dense-n1000-a1-dfa4 | 1 | 13.76 ms | 61.0 | 4.7 | 37% (DRAM) | 1.16 | 0.25 | 0 | compute/latency-bound |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | 1 | 2.17 ms | 0.8 | 0.4 | 1% (L2) | 0.07 | 0.20 | 8000 | compute/latency-bound |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | 1 | 22.51 ms | 7.8 | 0.4 | 1% (L2) | 0.07 | 0.20 | 80000 | compute/latency-bound |
| product-imdp-dense-n1000-a4-dfa4 | 1 | 72.74 ms | 244.1 | 3.5 | 28% (DRAM) | 0.88 | 0.25 | 0 | compute/latency-bound |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | 1 | 8.89 ms | 3.1 | 0.4 | 1% (L2) | 0.07 | 0.20 | 32000 | compute/latency-bound |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | 1 | 88.89 ms | 31.1 | 0.4 | 3% (DRAM) | 0.07 | 0.20 | 320000 | compute/latency-bound |
| imdp-dense-n100-a1 | 16 | 32.8 µs | 0.2 | 4.9 | 8% (L2) | 1.22 | 0.25 | 82 | overhead-bound |
| imdp-dense-n100-a4 | 16 | 36.2 µs | 0.6 | 17.7 | 30% (L2) | 4.42 | 0.25 | 82 | overhead-bound |
| imdp-dense-n1000-a1 | 16 | 764.8 µs | 15.3 | 20.9 | 36% (L2) | 5.23 | 0.25 | 82 | compute/latency-bound |
| imdp-dense-n1000-a4 | 16 | 2.89 ms | 61.0 | 22.2 | 51% (DRAM) | 5.54 | 0.25 | 82 | memory-bound |
| imdp-dense-n4000-a1 | 16 | 9.73 ms | 244.1 | 26.3 | 61% (DRAM) | 6.58 | 0.25 | 82 | memory-bound |
| imdp-dense-n4000-a4 | 16 | 41.18 ms | 976.6 | 24.9 | 57% (DRAM) | 6.22 | 0.25 | 82 | memory-bound |
| imdp-sparse-n10000-nnz10-a1 | 16 | 847.2 µs | 1.9 | 2.4 | 4% (L2) | 0.47 | 0.20 | 20082 | compute/latency-bound |
| imdp-sparse-n10000-nnz10-a4 | 16 | 3.16 ms | 7.8 | 2.6 | 4% (L2) | 0.51 | 0.20 | 80082 | compute/latency-bound |
| imdp-sparse-n10000-nnz100-a1 | 16 | 4.20 ms | 19.1 | 4.8 | 8% (L2) | 0.95 | 0.20 | 20082 | compute/latency-bound |
| imdp-sparse-n10000-nnz100-a4 | 16 | 16.62 ms | 76.4 | 4.8 | 11% (DRAM) | 0.96 | 0.20 | 80082 | compute/latency-bound |
| imdp-sparse-n100000-nnz10-a1 | 16 | 9.69 ms | 19.5 | 2.1 | 4% (L2) | 0.41 | 0.20 | 200082 | compute/latency-bound |
| imdp-sparse-n100000-nnz10-a4 | 16 | 43.56 ms | 77.8 | 1.9 | 4% (DRAM) | 0.37 | 0.20 | 800082 | compute/latency-bound |
| imdp-sparse-n100000-nnz100-a1 | 16 | 49.57 ms | 191.1 | 4.0 | 9% (DRAM) | 0.81 | 0.20 | 200082 | compute/latency-bound |
| imdp-sparse-n100000-nnz100-a4 | 16 | 192.83 ms | 764.5 | 4.2 | 10% (DRAM) | 0.83 | 0.20 | 800082 | compute/latency-bound |
| product-imdp-dense-n1000-a1-dfa4 | 16 | 3.06 ms | 61.0 | 20.9 | 48% (DRAM) | 5.24 | 0.25 | 328 | compute/latency-bound |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | 16 | 366.9 µs | 0.8 | 2.2 | 4% (L2) | 0.44 | 0.20 | 8328 | compute/latency-bound |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | 16 | 4.28 ms | 7.8 | 1.9 | 3% (L2) | 0.37 | 0.20 | 80328 | compute/latency-bound |
| product-imdp-dense-n1000-a4-dfa4 | 16 | 11.80 ms | 244.1 | 21.7 | 50% (DRAM) | 5.42 | 0.25 | 328 | compute/latency-bound |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | 16 | 2.16 ms | 3.1 | 1.5 | 3% (L2) | 0.30 | 0.20 | 32328 | compute/latency-bound |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | 16 | 10.41 ms | 31.1 | 3.1 | 7% (DRAM) | 0.61 | 0.20 | 320328 | compute/latency-bound |

AI = arithmetic intensity of the model (4 flop per 16–20 bytes ≈ 0.2–0.25 flop/B): far below the ridge point of this machine (in-L1 FMA roof / DRAM roof ≈ tens of flop/B), so the O-max kernel can only be memory-, latency- or overhead-bound, never FLOP-bound.

