# Run-to-run spread before the noise fix (baseline vs rerun1, compare.jl --spread)

## t = 1

| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |
|---|---|---|---|---|---:|---:|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | Float64 | 5/5 | 1.436 s / 1.444 s | 0.5% | 0.0% | ok |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | Float64 | 20/21 | 182.83 ms / 182.86 ms | 0.0% | 0.0% | ok |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_stationary | Float64 | 5/5 | 10.379 s / 10.516 s | 1.3% | 0.0% | ok |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | Float64 | 5/5 | 1.491 s / 1.522 s | 2.1% | 0.4% | ok |
| fimdp-dense-v2-d10-a1-mccormick | bellman | Float64 | 20/20 | 645.71 ms / 643.39 ms | 0.4% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | solve_rvi | Float64 | 3/3 | 55.738 s / 55.759 s | 0.0% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | workspace | Float64 | 100/96 | 450.91 µs / 465.56 µs | 3.2% | 0.1% | ok |
| fimdp-dense-v2-d10-a1-omax | bellman | Float64 | 2529/2384 | 545.72 µs / 546.20 µs | 0.1% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float64 | 76/74 | 51.28 ms / 50.89 ms | 0.8% | 1.1% | ok |
| fimdp-dense-v2-d10-a1-omax | workspace | Float64 | 65/73 | 2.74 µs / 2.63 µs | 4.1% | 0.3% | ok |
| fimdp-dense-v2-d10-a4-omax | bellman | Float64 | 483/581 | 2.15 ms / 2.15 ms | 0.2% | 0.0% | ok |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float64 | 24/24 | 168.54 ms / 166.01 ms | 1.5% | 0.6% | ok |
| fimdp-dense-v2-d10-a4-omax | workspace | Float64 | 10000/10000 | 4.48 µs / 4.51 µs | 0.7% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | bellman | Float64 | 20/20 | 151.41 ms / 149.94 ms | 1.0% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float64 | 5/5 | 13.258 s / 13.237 s | 0.2% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | workspace | Float64 | 2760/3064 | 65.40 µs / 64.92 µs | 0.7% | 0.0% | ok |
| fimdp-dense-v2-d50-a4-omax | bellman | Float64 | 20/20 | 607.22 ms / 608.17 ms | 0.2% | 0.0% | ok |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float64 | 3/3 | 50.231 s / 50.357 s | 0.3% | 0.0% | ok |
| fimdp-dense-v2-d50-a4-omax | workspace | Float64 | 968/968 | 400.49 µs / 399.80 µs | 0.2% | 0.0% | ok |
| fimdp-dense-v3-d10-a1-omax | bellman | Float64 | 33/34 | 57.95 ms / 57.86 ms | 0.1% | 0.0% | ok |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float64 | 5/5 | 6.095 s / 6.078 s | 0.3% | 0.0% | ok |
| fimdp-dense-v3-d10-a1-omax | workspace | Float64 | 61/62 | 26.73 µs / 29.24 µs | 9.4% | 0.1% | NOISY |
| fimdp-dense-v3-d10-a4-omax | bellman | Float64 | 20/20 | 232.25 ms / 237.66 ms | 2.3% | 0.4% | ok |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float64 | 4/4 | 20.689 s / 21.239 s | 2.7% | 0.3% | ok |
| fimdp-dense-v3-d10-a4-omax | workspace | Float64 | 3907/4073 | 44.64 µs / 43.13 µs | 3.5% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | bellman | Float64 | 20/20 | 95.45 ms / 98.58 ms | 3.3% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | solve_rvi | Float64 | 5/5 | 10.624 s / 10.461 s | 1.6% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | workspace | Float64 | 99/101 | 452.21 µs / 444.50 µs | 1.7% | 0.2% | ok |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | Float64 | 226/236 | 6.15 ms / 6.30 ms | 2.5% | 0.2% | ok |
| fimdp-sparse-v2-d10-k4-a1-vertex | solve_rvi | Float64 | 6/6 | 662.40 ms / 652.68 ms | 1.5% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | Float64 | 39/40 | 171 ns / 160 ns | 6.9% | 0.3% | NOISY |
| fimdp-sparse-v2-d10-k4-a4-mccormick | bellman | Float64 | 20/20 | 395.98 ms / 397.06 ms | 0.3% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | solve_rvi | Float64 | 4/4 | 19.753 s / 19.580 s | 0.9% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | workspace | Float64 | 570/590 | 402.60 µs / 405.61 µs | 0.7% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | bellman | Float64 | 63/65 | 25.55 ms / 25.62 ms | 0.2% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | Float64 | 5/5 | 1.359 s / 1.338 s | 1.6% | 0.4% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | Float64 | 10000/10000 | 94 ns / 91 ns | 2.8% | 0.2% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | bellman | Float64 | 93/101 | 18.02 ms / 17.94 ms | 0.5% | 0.0% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | solve_rvi | Float64 | 5/5 | 1.845 s / 1.805 s | 2.2% | 0.4% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | workspace | Float64 | 58/57 | 59.98 µs / 62.76 µs | 4.6% | 0.5% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | bellman | Float64 | 26/26 | 72.77 ms / 72.11 ms | 0.9% | 0.5% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | Float64 | 5/5 | 5.402 s / 5.386 s | 0.3% | 0.1% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | workspace | Float64 | 2197/2114 | 123.64 µs / 124.67 µs | 0.8% | 0.0% | ok |
| fimdp-sparse-v3-d10-k3-a1-mccormick | bellman | Float64 | 20/20 | 2.428 s / 2.425 s | 0.1% | 0.6% | ok |
| fimdp-sparse-v3-d10-k3-a1-mccormick | workspace | Float64 | 90/90 | 473.54 µs / 466.21 µs | 1.6% | 0.0% | ok |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | Float64 | 20/20 | 314.36 ms / 339.90 ms | 8.1% | 2.0% | NOISY |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | Float64 | 35/36 | 253 ns / 199 ns | 27.1% | 0.0% | NOISY |
| fimdp-sparse-v3-d20-k5-a1-omax | bellman | Float64 | 20/20 | 126.59 ms / 121.46 ms | 4.2% | 0.0% | ok |
| fimdp-sparse-v3-d20-k5-a1-omax | solve_rvi | Float64 | 5/5 | 14.069 s / 13.745 s | 2.4% | 0.3% | ok |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | Float64 | 54/57 | 153.19 µs / 187.09 µs | 22.1% | 0.4% | NOISY |
| fimdp-sparse-v3-d20-k5-a4-omax | bellman | Float64 | 20/20 | 506.75 ms / 499.58 ms | 1.4% | 0.0% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | Float64 | 3/3 | 40.221 s / 39.867 s | 0.9% | 0.5% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | workspace | Float64 | 793/816 | 489.42 µs / 478.07 µs | 2.4% | 0.0% | ok |
| imdp-dense-n100-a1 | bellman | Float64 | 10000/10000 | 34.97 µs / 34.94 µs | 0.1% | 0.4% | ok |
| imdp-dense-n100-a1 | solve_ivi | Float64 | 749/757 | 4.25 ms / 4.25 ms | 0.0% | 0.0% | ok |
| imdp-dense-n100-a1 | solve_rvi | Float64 | 1058/1072 | 2.91 ms / 2.92 ms | 0.3% | 0.2% | ok |
| imdp-dense-n100-a1 | workspace | Float64 | 75/75 | 4.07 µs / 5.38 µs | 32.3% | 0.3% | NOISY |
| imdp-dense-n100-a4 | bellman | Float64 | 7355/5320 | 138.11 µs / 138.24 µs | 0.1% | 0.0% | ok |
| imdp-dense-n100-a4 | solve_ivi | Float64 | 263/262 | 11.21 ms / 11.19 ms | 0.1% | 0.3% | ok |
| imdp-dense-n100-a4 | solve_rvi | Float64 | 292/289 | 9.97 ms / 9.97 ms | 0.0% | 0.3% | ok |
| imdp-dense-n100-a4 | workspace | Float64 | 10000/10000 | 9.37 µs / 9.45 µs | 0.8% | 0.0% | ok |
| imdp-dense-n1000-a1 | bellman | Float64 | 278/281 | 3.51 ms / 3.42 ms | 2.4% | 0.5% | ok |
| imdp-dense-n1000-a1 | solve_ivi | Float64 | 9/9 | 425.14 ms / 419.64 ms | 1.3% | 0.0% | ok |
| imdp-dense-n1000-a1 | solve_rvi | Float64 | 13/14 | 284.40 ms / 280.51 ms | 1.4% | 0.4% | ok |
| imdp-dense-n1000-a1 | workspace | Float64 | 1074/1033 | 366.21 µs / 366.52 µs | 0.1% | 0.0% | ok |
| imdp-dense-n1000-a4 | bellman | Float64 | 75/76 | 18.48 ms / 18.12 ms | 2.0% | 0.0% | ok |
| imdp-dense-n1000-a4 | solve_ivi | Float64 | 5/5 | 1.322 s / 1.326 s | 0.3% | 0.0% | ok |
| imdp-dense-n1000-a4 | solve_rvi | Float64 | 5/5 | 1.432 s / 1.436 s | 0.3% | 0.0% | ok |
| imdp-dense-n1000-a4 | workspace | Float64 | 275/272 | 5.59 ms / 4.09 ms | 36.5% | 0.3% | NOISY |
| imdp-dense-n4000-a1 | bellman | Float64 | 23/23 | 78.04 ms / 77.92 ms | 0.2% | 0.0% | ok |
| imdp-dense-n4000-a1 | solve_ivi | Float64 | 5/5 | 8.583 s / 8.651 s | 0.8% | 0.0% | ok |
| imdp-dense-n4000-a1 | solve_rvi | Float64 | 5/5 | 6.259 s / 6.296 s | 0.6% | 0.2% | ok |
| imdp-dense-n4000-a1 | workspace | Float64 | 71/71 | 16.91 ms / 17.23 ms | 1.9% | 0.0% | ok |
| imdp-dense-n4000-a4 | bellman | Float64 | 20/20 | 312.53 ms / 311.86 ms | 0.2% | 0.0% | ok |
| imdp-dense-n4000-a4 | solve_ivi | Float64 | 4/4 | 22.101 s / 22.187 s | 0.4% | 0.4% | ok |
| imdp-dense-n4000-a4 | solve_rvi | Float64 | 3/3 | 24.900 s / 24.907 s | 0.0% | 0.3% | ok |
| imdp-dense-n4000-a4 | workspace | Float64 | 16/16 | 70.08 ms / 70.85 ms | 1.1% | 0.3% | ok |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float64 | 244/258 | 6.31 ms / 6.27 ms | 0.6% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a1 | solve_ivi | Float64 | 5/5 | 774.08 ms / 781.95 ms | 1.0% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float64 | 7/7 | 614.63 ms / 608.64 ms | 1.0% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float64 | 63/70 | 131.53 µs / 173.38 µs | 31.8% | 0.5% | NOISY |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float64 | 78/66 | 25.22 ms / 24.81 ms | 1.7% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | solve_ivi | Float64 | 5/5 | 3.694 s / 3.682 s | 0.3% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float64 | 5/5 | 1.156 s / 1.135 s | 1.9% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float64 | 1323/1350 | 271.10 µs / 269.43 µs | 0.6% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float64 | 47/44 | 41.09 ms / 41.56 ms | 1.1% | 28.2% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | solve_ivi | Float64 | 5/5 | 4.999 s / 5.030 s | 0.6% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 3.314 s / 3.374 s | 1.8% | 0.3% | ok |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float64 | 960/982 | 378.53 µs / 377.82 µs | 0.2% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float64 | 20/20 | 165.76 ms / 168.28 ms | 1.5% | 0.2% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_ivi | Float64 | 5/5 | 14.420 s / 14.529 s | 0.8% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 10.540 s / 10.776 s | 2.2% | 0.3% | ok |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float64 | 278/274 | 2.92 ms / 2.93 ms | 0.6% | 0.1% | ok |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float64 | 29/28 | 64.79 ms / 63.59 ms | 1.9% | 0.5% | ok |
| imdp-sparse-n100000-nnz10-a1 | solve_ivi | Float64 | 5/5 | 8.055 s / 8.021 s | 0.4% | 135.7% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float64 | 5/5 | 6.188 s / 6.103 s | 1.4% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float64 | 463/565 | 709.38 µs / 678.29 µs | 4.6% | 0.4% | ok |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float64 | 20/20 | 267.02 ms / 267.83 ms | 0.3% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a4 | solve_ivi | Float64 | 3/3 | 38.635 s / 38.244 s | 1.0% | 0.9% | ok |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float64 | 5/5 | 11.436 s / 11.680 s | 2.1% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float64 | 169/168 | 4.79 ms / 4.68 ms | 2.3% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float64 | 20/20 | 432.05 ms / 432.14 ms | 0.0% | 0.3% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_ivi | Float64 | 3/3 | 51.892 s / 52.410 s | 1.0% | 0.3% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float64 | 3/3 | 34.583 s / 34.974 s | 1.1% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float64 | 112/114 | 8.26 ms / 8.16 ms | 1.2% | 0.4% | ok |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float64 | 20/20 | 1.736 s / 1.744 s | 0.5% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | solve_ivi | Float64 | 3/3 | 148.794 s / 150.000 s | 0.8% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float64 | 3/3 | 109.089 s / 110.602 s | 1.4% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float64 | 28/28 | 38.15 ms / 38.19 ms | 0.1% | 0.2% | ok |
| product-imdp-dense-n1000-a1-dfa4 | bellman | Float64 | 91/93 | 13.76 ms / 13.73 ms | 0.2% | 0.3% | ok |
| product-imdp-dense-n1000-a1-dfa4 | solve_dfa | Float64 | 5/5 | 2.013 s / 2.004 s | 0.4% | 0.3% | ok |
| product-imdp-dense-n1000-a1-dfa4 | workspace | Float64 | 42/43 | 473.41 µs / 470.96 µs | 0.5% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | bellman | Float64 | 23/23 | 72.74 ms / 72.53 ms | 0.3% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | Float64 | 5/5 | 8.953 s / 8.914 s | 0.4% | 0.4% | ok |
| product-imdp-dense-n1000-a4-dfa4 | workspace | Float64 | 278/267 | 5.60 ms / 5.67 ms | 1.2% | 0.0% | ok |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | Float64 | 618/599 | 2.17 ms / 2.18 ms | 0.5% | 0.0% | ok |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | solve_dfa | Float64 | 12/12 | 338.97 ms / 340.80 ms | 0.5% | 0.7% | ok |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | Float64 | 40/40 | 15.57 µs / 13.80 µs | 12.8% | 0.0% | NOISY |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | Float64 | 127/165 | 8.89 ms / 8.91 ms | 0.3% | 0.3% | ok |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | Float64 | 9/9 | 458.03 ms / 458.43 ms | 0.1% | 0.0% | ok |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | workspace | Float64 | 7560/7433 | 25.85 µs / 25.91 µs | 0.2% | 0.0% | ok |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | Float64 | 72/71 | 22.51 ms / 22.44 ms | 0.3% | 0.2% | ok |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | Float64 | 5/5 | 3.368 s / 3.358 s | 0.3% | 0.4% | ok |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | Float64 | 3130/3734 | 62.34 µs / 62.00 µs | 0.6% | 0.0% | ok |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | Float64 | 20/20 | 88.89 ms / 89.65 ms | 0.9% | 0.0% | ok |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | Float64 | 5/5 | 4.433 s / 4.528 s | 2.1% | 0.0% | ok |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | Float64 | 1343/1214 | 273.47 µs / 274.65 µs | 0.4% | 0.0% | ok |
| real-multiObj_robotIMDP | bellman | Float64 | 3876/2830 | 304.90 µs / 311.57 µs | 2.2% | 0.5% | ok |
| real-multiObj_robotIMDP | solve_cs_stationary | Float64 | 142/127 | 24.85 ms / 24.96 ms | 0.4% | 0.0% | ok |
| real-multiObj_robotIMDP | solve_rvi | Float64 | 135/137 | 24.74 ms / 25.11 ms | 1.5% | 0.0% | ok |
| real-multiObj_robotIMDP | workspace | Float64 | 57/57 | 8.98 µs / 9.02 µs | 0.4% | 0.0% | ok |

131 entries, 9 with spread > 5.0%.
spread: median 0.82%, 90th percentile 3.50%, max 36.55%

## t = 4

| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |
|---|---|---|---|---|---:|---:|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | Float64 | 8/8 | 493.60 ms / 485.54 ms | 1.7% | 0.3% | ok |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | Float64 | 54/55 | 62.28 ms / 60.98 ms | 2.1% | 0.1% | ok |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_stationary | Float64 | 5/5 | 2.784 s / 2.787 s | 0.1% | 0.0% | ok |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | Float64 | 10/10 | 427.66 ms / 416.29 ms | 2.7% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | bellman | Float64 | 20/20 | 172.97 ms / 173.04 ms | 0.0% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | solve_rvi | Float64 | 5/5 | 14.919 s / 14.949 s | 0.2% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | workspace | Float64 | 109/104 | 1.61 ms / 1.63 ms | 1.1% | 0.4% | ok |
| fimdp-dense-v2-d10-a1-omax | bellman | Float64 | 2896/3648 | 409.78 µs / 409.93 µs | 0.0% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float64 | 59/56 | 65.99 ms / 65.81 ms | 0.3% | 0.5% | ok |
| fimdp-dense-v2-d10-a1-omax | workspace | Float64 | 76/76 | 6.51 µs / 6.91 µs | 6.1% | 0.2% | NOISY |
| fimdp-dense-v2-d10-a4-omax | bellman | Float64 | 1152/528 | 1.02 ms / 1.05 ms | 3.1% | 0.5% | ok |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float64 | 37/36 | 120.15 ms / 121.52 ms | 1.1% | 0.3% | ok |
| fimdp-dense-v2-d10-a4-omax | workspace | Float64 | 10000/10000 | 17.57 µs / 17.42 µs | 0.9% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | bellman | Float64 | 39/40 | 44.34 ms / 43.18 ms | 2.7% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float64 | 5/5 | 3.783 s / 3.737 s | 1.2% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | workspace | Float64 | 1524/1571 | 261.87 µs / 254.28 µs | 3.0% | 1.2% | ok |
| fimdp-dense-v2-d50-a4-omax | bellman | Float64 | 20/20 | 164.37 ms / 164.49 ms | 0.1% | 0.0% | ok |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float64 | 5/5 | 13.702 s / 13.680 s | 0.2% | 0.3% | ok |
| fimdp-dense-v2-d50-a4-omax | workspace | Float64 | 391/374 | 1.59 ms / 1.59 ms | 0.2% | 0.6% | ok |
| fimdp-dense-v3-d10-a1-omax | bellman | Float64 | 105/106 | 17.58 ms / 15.92 ms | 10.4% | 0.2% | NOISY |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float64 | 5/5 | 1.700 s / 1.689 s | 0.7% | 0.4% | ok |
| fimdp-dense-v3-d10-a1-omax | workspace | Float64 | 72/73 | 79.70 µs / 81.83 µs | 2.7% | 0.2% | ok |
| fimdp-dense-v3-d10-a4-omax | bellman | Float64 | 29/30 | 62.87 ms / 62.33 ms | 0.9% | 0.0% | ok |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float64 | 5/5 | 5.736 s / 5.637 s | 1.7% | 0.0% | ok |
| fimdp-dense-v3-d10-a4-omax | workspace | Float64 | 2170/1902 | 197.48 µs / 199.21 µs | 0.9% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | bellman | Float64 | 61/61 | 28.27 ms / 27.86 ms | 1.5% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | solve_rvi | Float64 | 5/5 | 3.063 s / 3.024 s | 1.3% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | workspace | Float64 | 105/105 | 1.62 ms / 1.63 ms | 0.7% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | Float64 | 840/318 | 2.02 ms / 2.88 ms | 42.7% | 27.8% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | solve_rvi | Float64 | 12/12 | 361.55 ms / 355.31 ms | 1.8% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | Float64 | 48/49 | 406 ns / 430 ns | 5.9% | 0.3% | NOISY |
| fimdp-sparse-v2-d10-k4-a4-mccormick | bellman | Float64 | 20/20 | 110.84 ms / 110.55 ms | 0.3% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | solve_rvi | Float64 | 5/5 | 5.315 s / 5.251 s | 1.2% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | workspace | Float64 | 259/284 | 1.60 ms / 1.61 ms | 0.6% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | bellman | Float64 | 185/175 | 11.69 ms / 11.36 ms | 2.9% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | Float64 | 8/8 | 537.80 ms / 537.95 ms | 0.0% | 0.8% | ok |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | Float64 | 10000/10000 | 378 ns / 455 ns | 20.5% | 0.2% | NOISY |
| fimdp-sparse-v2-d50-k10-a1-omax | bellman | Float64 | 187/199 | 5.56 ms / 5.78 ms | 4.0% | 0.0% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | solve_rvi | Float64 | 6/7 | 625.28 ms / 612.07 ms | 2.2% | 0.3% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | workspace | Float64 | 69/70 | 158.70 µs / 160.03 µs | 0.8% | 0.0% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | bellman | Float64 | 85/86 | 19.77 ms / 19.63 ms | 0.7% | 0.0% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | Float64 | 5/5 | 1.486 s / 1.474 s | 0.8% | 0.0% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | workspace | Float64 | 855/958 | 510.13 µs / 502.94 µs | 1.4% | 0.0% | ok |
| fimdp-sparse-v3-d10-k3-a1-mccormick | bellman | Float64 | 20/20 | 636.86 ms / 633.77 ms | 0.5% | 0.4% | ok |
| fimdp-sparse-v3-d10-k3-a1-mccormick | workspace | Float64 | 102/94 | 1.62 ms / 1.64 ms | 1.4% | 0.0% | ok |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | Float64 | 22/21 | 83.58 ms / 88.93 ms | 6.4% | 0.0% | NOISY |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | Float64 | 44/43 | 626 ns / 733 ns | 17.0% | 0.3% | NOISY |
| fimdp-sparse-v3-d20-k5-a1-omax | bellman | Float64 | 50/50 | 33.69 ms / 33.82 ms | 0.4% | 0.2% | ok |
| fimdp-sparse-v3-d20-k5-a1-omax | solve_rvi | Float64 | 5/5 | 3.850 s / 3.752 s | 2.6% | 0.0% | ok |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | Float64 | 64/63 | 540.64 µs / 536.04 µs | 0.9% | 0.2% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | bellman | Float64 | 20/20 | 137.31 ms / 131.94 ms | 4.1% | 0.3% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | Float64 | 5/5 | 10.807 s / 10.620 s | 1.8% | 0.0% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | workspace | Float64 | 304/312 | 2.07 ms / 2.06 ms | 0.2% | 0.2% | ok |
| imdp-dense-n100-a1 | bellman | Float64 | 3854/7033 | 24.09 µs / 11.01 µs | 118.8% | 28.2% | NOISY (clock deviating) |
| imdp-dense-n100-a1 | solve_ivi | Float64 | 601/490 | 2.43 ms / 2.79 ms | 14.7% | 0.0% | NOISY |
| imdp-dense-n100-a1 | solve_rvi | Float64 | 700/580 | 956.25 µs / 973.76 µs | 1.8% | 0.3% | ok |
| imdp-dense-n100-a1 | workspace | Float64 | 89/90 | 3.25 µs / 2.90 µs | 12.2% | 0.3% | NOISY |
| imdp-dense-n100-a4 | bellman | Float64 | 5251/7734 | 33.24 µs / 33.11 µs | 0.4% | 62.1% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_ivi | Float64 | 503/462 | 5.23 ms / 5.15 ms | 1.5% | 28.2% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_rvi | Float64 | 595/773 | 4.49 ms / 4.51 ms | 0.4% | 0.0% | ok (clock deviating) |
| imdp-dense-n100-a4 | workspace | Float64 | 10000/10000 | 4.79 µs / 4.77 µs | 0.4% | 0.0% | ok (clock deviating) |
| imdp-dense-n1000-a1 | bellman | Float64 | 1007/333 | 1.36 ms / 1.41 ms | 3.5% | 0.2% | ok |
| imdp-dense-n1000-a1 | solve_ivi | Float64 | 16/16 | 252.67 ms / 252.12 ms | 0.2% | 0.4% | ok |
| imdp-dense-n1000-a1 | solve_rvi | Float64 | 25/25 | 167.61 ms / 167.10 ms | 0.3% | 0.0% | ok |
| imdp-dense-n1000-a1 | workspace | Float64 | 1270/1392 | 365.10 µs / 166.81 µs | 118.9% | 27.7% | NOISY (clock deviating) |
| imdp-dense-n1000-a4 | bellman | Float64 | 304/292 | 5.20 ms / 5.28 ms | 1.7% | 0.0% | ok |
| imdp-dense-n1000-a4 | solve_ivi | Float64 | 7/7 | 570.01 ms / 571.27 ms | 0.2% | 0.5% | ok |
| imdp-dense-n1000-a4 | solve_rvi | Float64 | 8/8 | 482.93 ms / 504.18 ms | 4.4% | 0.0% | ok |
| imdp-dense-n1000-a4 | workspace | Float64 | 343/340 | 5.46 ms / 5.36 ms | 2.0% | 0.0% | ok |
| imdp-dense-n4000-a1 | bellman | Float64 | 65/63 | 25.47 ms / 25.47 ms | 0.0% | 0.0% | ok |
| imdp-dense-n4000-a1 | solve_ivi | Float64 | 5/5 | 2.666 s / 2.694 s | 1.1% | 15.0% | ok (clock deviating) |
| imdp-dense-n4000-a1 | solve_rvi | Float64 | 5/5 | 1.955 s / 1.993 s | 2.0% | 0.0% | ok |
| imdp-dense-n4000-a1 | workspace | Float64 | 83/82 | 17.09 ms / 17.15 ms | 0.3% | 0.0% | ok |
| imdp-dense-n4000-a4 | bellman | Float64 | 20/20 | 93.68 ms / 95.92 ms | 2.4% | 0.0% | ok |
| imdp-dense-n4000-a4 | solve_ivi | Float64 | 5/5 | 6.632 s / 6.576 s | 0.9% | 0.5% | ok |
| imdp-dense-n4000-a4 | solve_rvi | Float64 | 5/5 | 7.407 s / 7.559 s | 2.0% | 0.3% | ok |
| imdp-dense-n4000-a4 | workspace | Float64 | 17/18 | 69.97 ms / 71.09 ms | 1.6% | 27.7% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float64 | 815/741 | 2.23 ms / 2.16 ms | 3.3% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a1 | solve_ivi | Float64 | 11/11 | 389.73 ms / 395.87 ms | 1.6% | 0.3% | ok |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float64 | 14/14 | 312.11 ms / 304.68 ms | 2.4% | 0.5% | ok |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float64 | 73/73 | 642.38 µs / 632.48 µs | 1.6% | 0.1% | ok |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float64 | 203/236 | 8.30 ms / 7.01 ms | 18.4% | 0.0% | NOISY |
| imdp-sparse-n10000-nnz10-a4 | solve_ivi | Float64 | 5/5 | 1.364 s / 1.351 s | 1.0% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float64 | 10/11 | 363.54 ms / 356.10 ms | 2.1% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float64 | 557/542 | 1.06 ms / 1.08 ms | 1.8% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float64 | 146/101 | 13.38 ms / 10.93 ms | 22.4% | 0.3% | NOISY |
| imdp-sparse-n10000-nnz100-a1 | solve_ivi | Float64 | 5/5 | 1.523 s / 1.450 s | 5.1% | 0.0% | NOISY |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 958.58 ms / 978.34 ms | 2.1% | 1.5% | ok |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float64 | 434/412 | 1.54 ms / 1.56 ms | 1.5% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float64 | 38/38 | 45.01 ms / 42.99 ms | 4.7% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_ivi | Float64 | 5/5 | 3.895 s / 3.950 s | 1.4% | 0.3% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 2.845 s / 2.799 s | 1.6% | 1.6% | ok |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float64 | 87/89 | 11.52 ms / 11.77 ms | 2.2% | 0.3% | ok |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float64 | 97/97 | 16.82 ms / 17.22 ms | 2.3% | 0.3% | ok |
| imdp-sparse-n100000-nnz10-a1 | solve_ivi | Float64 | 5/5 | 2.356 s / 2.288 s | 2.9% | 1.4% | ok |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float64 | 5/5 | 1.816 s / 1.831 s | 0.8% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float64 | 236/246 | 3.02 ms / 2.94 ms | 2.8% | 0.9% | ok |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float64 | 27/31 | 70.29 ms / 68.93 ms | 2.0% | 0.2% | ok |
| imdp-sparse-n100000-nnz10-a4 | solve_ivi | Float64 | 5/5 | 10.164 s / 10.129 s | 0.3% | 0.3% | ok |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float64 | 5/5 | 3.207 s / 3.171 s | 1.1% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float64 | 53/52 | 19.57 ms / 19.89 ms | 1.6% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float64 | 20/20 | 112.19 ms / 111.89 ms | 0.3% | 0.4% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_ivi | Float64 | 5/5 | 13.608 s / 13.727 s | 0.9% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 9.141 s / 9.080 s | 0.7% | 0.4% | ok |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float64 | 33/32 | 34.78 ms / 32.30 ms | 7.7% | 0.3% | NOISY |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float64 | 20/20 | 404.19 ms / 439.39 ms | 8.7% | 0.0% | NOISY |
| imdp-sparse-n100000-nnz100-a4 | solve_ivi | Float64 | 3/3 | 38.497 s / 38.866 s | 1.0% | 0.4% | ok |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float64 | 3/3 | 28.100 s / 28.151 s | 0.2% | 0.4% | ok |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float64 | 10/10 | 151.36 ms / 153.76 ms | 1.6% | 0.0% | ok |
| product-imdp-dense-n1000-a1-dfa4 | bellman | Float64 | 198/283 | 8.21 ms / 8.21 ms | 0.0% | 0.3% | ok |
| product-imdp-dense-n1000-a1-dfa4 | solve_dfa | Float64 | 5/5 | 1.185 s / 1.198 s | 1.1% | 0.2% | ok |
| product-imdp-dense-n1000-a1-dfa4 | workspace | Float64 | 56/56 | 421.64 µs / 402.87 µs | 4.7% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | bellman | Float64 | 68/68 | 23.27 ms / 24.02 ms | 3.2% | 0.4% | ok |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | Float64 | 5/5 | 3.122 s / 3.128 s | 0.2% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | workspace | Float64 | 323/162 | 5.59 ms / 5.33 ms | 4.9% | 27.3% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | Float64 | 426/409 | 1.98 ms / 1.85 ms | 7.1% | 0.5% | NOISY |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | solve_dfa | Float64 | 11/9 | 446.08 ms / 437.01 ms | 2.1% | 28.2% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | Float64 | 53/53 | 41.38 µs / 36.48 µs | 13.4% | 0.2% | NOISY |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | Float64 | 307/266 | 6.85 ms / 6.87 ms | 0.3% | 0.1% | ok |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | Float64 | 13/18 | 330.58 ms / 213.43 ms | 54.9% | 27.8% | NOISY (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | workspace | Float64 | 3766/3837 | 101.81 µs / 101.83 µs | 0.0% | 0.3% | ok |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | Float64 | 149/257 | 12.77 ms / 11.78 ms | 8.4% | 0.3% | NOISY |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | Float64 | 5/5 | 1.789 s / 1.800 s | 0.6% | 0.3% | ok |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | Float64 | 2001/1749 | 242.42 µs / 242.03 µs | 0.2% | 0.0% | ok |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | Float64 | 65/66 | 31.15 ms / 29.41 ms | 5.9% | 0.0% | NOISY |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | Float64 | 5/5 | 1.546 s / 1.400 s | 10.4% | 0.0% | NOISY |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | Float64 | 513/541 | 1.09 ms / 1.09 ms | 0.3% | 0.4% | ok |
| real-multiObj_robotIMDP | bellman | Float64 | 5687/6200 | 198.41 µs / 196.81 µs | 0.8% | 0.0% | ok |
| real-multiObj_robotIMDP | solve_cs_stationary | Float64 | 153/184 | 15.80 ms / 16.51 ms | 4.5% | 0.0% | ok |
| real-multiObj_robotIMDP | solve_rvi | Float64 | 117/150 | 15.79 ms / 15.25 ms | 3.6% | 0.6% | ok |
| real-multiObj_robotIMDP | workspace | Float64 | 74/75 | 27.11 µs / 25.98 µs | 4.3% | 0.2% | ok |

131 entries, 22 with spread > 5.0%.
spread: median 1.59%, 90th percentile 8.71%, max 118.88%

## t = 8

| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |
|---|---|---|---|---|---:|---:|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | Float64 | 18/14 | 247.83 ms / 231.17 ms | 7.2% | 0.4% | NOISY (clock deviating) |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | Float64 | 78/78 | 42.02 ms / 31.09 ms | 35.2% | 0.0% | NOISY |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_stationary | Float64 | 5/5 | 2.064 s / 2.041 s | 1.2% | 0.0% | ok |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | Float64 | 13/25 | 312.08 ms / 309.09 ms | 1.0% | 27.8% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-mccormick | bellman | Float64 | 20/20 | 128.83 ms / 123.00 ms | 4.7% | 0.0% | ok |
| fimdp-dense-v2-d10-a1-mccormick | solve_rvi | Float64 | 5/5 | 11.176 s / 11.293 s | 1.0% | 0.1% | ok |
| fimdp-dense-v2-d10-a1-mccormick | workspace | Float64 | 117/90 | 1.51 ms / 3.23 ms | 114.7% | 27.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | bellman | Float64 | 10000/618 | 67.97 µs / 152.09 µs | 123.7% | 27.9% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float64 | 129/109 | 10.71 ms / 10.74 ms | 0.3% | 28.2% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | workspace | Float64 | 76/76 | 8.80 µs / 12.04 µs | 36.8% | 27.8% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a4-omax | bellman | Float64 | 2031/1824 | 632.18 µs / 619.91 µs | 2.0% | 0.2% | ok |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float64 | 62/65 | 65.45 ms / 65.73 ms | 0.4% | 0.1% | ok (clock deviating) |
| fimdp-dense-v2-d10-a4-omax | workspace | Float64 | 10000/10000 | 20.29 µs / 22.89 µs | 12.8% | 0.0% | NOISY (clock deviating) |
| fimdp-dense-v2-d50-a1-omax | bellman | Float64 | 63/53 | 31.99 ms / 32.87 ms | 2.7% | 0.2% | ok |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float64 | 5/5 | 2.883 s / 2.872 s | 0.4% | 0.3% | ok |
| fimdp-dense-v2-d50-a1-omax | workspace | Float64 | 1253/1254 | 264.96 µs / 269.24 µs | 1.6% | 0.3% | ok (clock deviating) |
| fimdp-dense-v2-d50-a4-omax | bellman | Float64 | 20/20 | 125.77 ms / 128.69 ms | 2.3% | 0.0% | ok |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float64 | 5/5 | 10.740 s / 10.821 s | 0.8% | 0.3% | ok |
| fimdp-dense-v2-d50-a4-omax | workspace | Float64 | 268/240 | 3.18 ms / 1.54 ms | 106.4% | 0.0% | NOISY (clock deviating) |
| fimdp-dense-v3-d10-a1-omax | bellman | Float64 | 139/136 | 11.18 ms / 4.62 ms | 142.0% | 27.9% | NOISY (clock deviating) |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float64 | 5/6 | 1.222 s / 1.226 s | 0.3% | 0.3% | ok |
| fimdp-dense-v3-d10-a1-omax | workspace | Float64 | 72/72 | 134.52 µs / 123.11 µs | 9.3% | 0.0% | NOISY |
| fimdp-dense-v3-d10-a4-omax | bellman | Float64 | 70/46 | 47.32 ms / 45.95 ms | 3.0% | 28.2% | ok (clock deviating) |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float64 | 5/5 | 4.294 s / 4.275 s | 0.4% | 0.0% | ok |
| fimdp-dense-v3-d10-a4-omax | workspace | Float64 | 1448/1382 | 393.64 µs / 274.13 µs | 43.6% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-mccormick | bellman | Float64 | 103/104 | 19.96 ms / 20.00 ms | 0.2% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | solve_rvi | Float64 | 5/5 | 2.224 s / 2.237 s | 0.6% | 0.3% | ok |
| fimdp-sparse-v2-d10-k4-a1-mccormick | workspace | Float64 | 109/109 | 1.50 ms / 1.48 ms | 1.1% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | Float64 | 941/1059 | 1.23 ms / 1.08 ms | 14.0% | 0.0% | NOISY |
| fimdp-sparse-v2-d10-k4-a1-vertex | solve_rvi | Float64 | 26/30 | 145.21 ms / 147.29 ms | 1.4% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | Float64 | 72/62 | 834 ns / 715 ns | 16.6% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-mccormick | bellman | Float64 | 24/26 | 85.20 ms / 84.54 ms | 0.8% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | solve_rvi | Float64 | 5/5 | 4.099 s / 4.134 s | 0.9% | 0.0% | ok |
| fimdp-sparse-v2-d10-k4-a4-mccormick | workspace | Float64 | 200/183 | 1.50 ms / 1.49 ms | 1.2% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | bellman | Float64 | 365/310 | 3.15 ms / 3.25 ms | 3.2% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | Float64 | 25/11 | 187.01 ms / 360.89 ms | 93.0% | 27.8% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | Float64 | 10000/10000 | 899 ns / 889 ns | 1.1% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d50-k10-a1-omax | bellman | Float64 | 382/515 | 3.56 ms / 1.82 ms | 95.0% | 27.6% | NOISY (clock deviating) |
| fimdp-sparse-v2-d50-k10-a1-omax | solve_rvi | Float64 | 10/10 | 389.05 ms / 382.45 ms | 1.7% | 0.1% | ok |
| fimdp-sparse-v2-d50-k10-a1-omax | workspace | Float64 | 67/69 | 274.80 µs / 194.20 µs | 41.5% | 27.9% | NOISY (clock deviating) |
| fimdp-sparse-v2-d50-k10-a4-omax | bellman | Float64 | 133/140 | 14.11 ms / 14.04 ms | 0.5% | 0.4% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | Float64 | 5/5 | 1.064 s / 1.057 s | 0.7% | 0.4% | ok |
| fimdp-sparse-v2-d50-k10-a4-omax | workspace | Float64 | 599/652 | 989.34 µs / 599.75 µs | 65.0% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-mccormick | bellman | Float64 | 20/20 | 483.82 ms / 487.37 ms | 0.7% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-mccormick | workspace | Float64 | 103/104 | 1.51 ms / 1.49 ms | 1.8% | 15.5% | ok (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | Float64 | 22/22 | 89.48 ms / 89.97 ms | 0.5% | 0.0% | ok |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | Float64 | 44/57 | 1.26 µs / 1.08 µs | 16.4% | 27.6% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a1-omax | bellman | Float64 | 75/73 | 24.50 ms / 24.60 ms | 0.4% | 0.3% | ok |
| fimdp-sparse-v3-d20-k5-a1-omax | solve_rvi | Float64 | 5/5 | 2.774 s / 2.808 s | 1.2% | 27.7% | ok (clock deviating) |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | Float64 | 88/60 | 937.97 µs / 1.65 ms | 75.9% | 26.3% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a4-omax | bellman | Float64 | 22/21 | 99.07 ms / 99.23 ms | 0.2% | 0.0% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | Float64 | 5/5 | 7.990 s / 8.037 s | 0.6% | 0.6% | ok |
| fimdp-sparse-v3-d20-k5-a4-omax | workspace | Float64 | 233/224 | 4.13 ms / 4.03 ms | 2.5% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a1 | bellman | Float64 | 9032/5860 | 7.83 µs / 14.49 µs | 85.2% | 0.1% | NOISY |
| imdp-dense-n100-a1 | solve_ivi | Float64 | 664/713 | 1.10 ms / 1.10 ms | 0.4% | 0.0% | ok |
| imdp-dense-n100-a1 | solve_rvi | Float64 | 743/732 | 765.84 µs / 772.96 µs | 0.9% | 0.3% | ok |
| imdp-dense-n100-a1 | workspace | Float64 | 89/72 | 2.41 µs / 2.53 µs | 4.8% | 0.0% | ok (clock deviating) |
| imdp-dense-n100-a4 | bellman | Float64 | 6446/6233 | 19.42 µs / 19.13 µs | 1.5% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_ivi | Float64 | 599/490 | 4.36 ms / 4.36 ms | 0.1% | 0.4% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_rvi | Float64 | 1998/1199 | 2.75 ms / 2.87 ms | 4.4% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a4 | workspace | Float64 | 10000/10000 | 4.70 µs / 4.95 µs | 5.4% | 0.3% | NOISY (clock deviating) |
| imdp-dense-n1000-a1 | bellman | Float64 | 1701/2216 | 789.69 µs / 751.43 µs | 5.1% | 27.8% | NOISY (clock deviating) |
| imdp-dense-n1000-a1 | solve_ivi | Float64 | 32/34 | 123.45 ms / 124.67 ms | 1.0% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a1 | solve_rvi | Float64 | 51/56 | 78.10 ms / 81.39 ms | 4.2% | 0.0% | ok (clock deviating) |
| imdp-dense-n1000-a1 | workspace | Float64 | 1480/1452 | 166.76 µs / 166.80 µs | 0.0% | 0.0% | ok (clock deviating) |
| imdp-dense-n1000-a4 | bellman | Float64 | 451/417 | 4.12 ms / 4.02 ms | 2.4% | 1.4% | ok |
| imdp-dense-n1000-a4 | solve_ivi | Float64 | 14/15 | 256.78 ms / 252.08 ms | 1.9% | 0.0% | ok (clock deviating) |
| imdp-dense-n1000-a4 | solve_rvi | Float64 | 14/12 | 310.69 ms / 309.67 ms | 0.3% | 0.4% | ok |
| imdp-dense-n1000-a4 | workspace | Float64 | 347/369 | 5.66 ms / 5.66 ms | 0.0% | 0.0% | ok (clock deviating) |
| imdp-dense-n4000-a1 | bellman | Float64 | 102/98 | 18.56 ms / 18.57 ms | 0.1% | 0.4% | ok |
| imdp-dense-n4000-a1 | solve_ivi | Float64 | 5/5 | 1.433 s / 1.427 s | 0.4% | 0.5% | ok |
| imdp-dense-n4000-a1 | solve_rvi | Float64 | 5/5 | 1.517 s / 1.520 s | 0.2% | 0.0% | ok |
| imdp-dense-n4000-a1 | workspace | Float64 | 87/85 | 15.98 ms / 15.25 ms | 4.8% | 0.3% | ok |
| imdp-dense-n4000-a4 | bellman | Float64 | 29/30 | 66.68 ms / 66.21 ms | 0.7% | 0.1% | ok |
| imdp-dense-n4000-a4 | solve_ivi | Float64 | 5/5 | 3.364 s / 3.369 s | 0.1% | 0.0% | ok |
| imdp-dense-n4000-a4 | solve_rvi | Float64 | 5/5 | 5.343 s / 5.352 s | 0.2% | 0.0% | ok |
| imdp-dense-n4000-a4 | workspace | Float64 | 19/16 | 71.15 ms / 72.87 ms | 2.4% | 28.2% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float64 | 986/917 | 1.03 ms / 1.01 ms | 2.3% | 0.4% | ok |
| imdp-sparse-n10000-nnz10-a1 | solve_ivi | Float64 | 25/22 | 164.10 ms / 165.81 ms | 1.0% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float64 | 29/29 | 128.64 ms / 127.90 ms | 0.6% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float64 | 70/72 | 1.28 ms / 1.28 ms | 0.7% | 0.2% | ok |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float64 | 305/314 | 2.29 ms / 4.98 ms | 118.0% | 0.0% | NOISY |
| imdp-sparse-n10000-nnz10-a4 | solve_ivi | Float64 | 7/7 | 545.74 ms / 533.86 ms | 2.2% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float64 | 25/16 | 123.66 ms / 124.81 ms | 0.9% | 27.7% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float64 | 462/419 | 2.16 ms / 2.17 ms | 0.7% | 0.1% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float64 | 204/183 | 8.23 ms / 8.25 ms | 0.3% | 0.3% | ok |
| imdp-sparse-n10000-nnz100-a1 | solve_ivi | Float64 | 9/5 | 1.017 s / 1.006 s | 1.1% | 28.2% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float64 | 6/6 | 682.24 ms / 684.77 ms | 0.4% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float64 | 294/217 | 1.89 ms / 1.91 ms | 0.8% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float64 | 62/62 | 32.36 ms / 32.27 ms | 0.3% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_ivi | Float64 | 5/5 | 2.839 s / 2.847 s | 0.3% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 2.100 s / 2.087 s | 0.6% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float64 | 49/46 | 21.68 ms / 23.20 ms | 7.0% | 0.3% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float64 | 143/112 | 12.46 ms / 12.71 ms | 2.0% | 0.9% | ok |
| imdp-sparse-n100000-nnz10-a1 | solve_ivi | Float64 | 5/5 | 752.77 ms / 767.83 ms | 2.0% | 0.3% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float64 | 5/5 | 1.265 s / 564.60 ms | 124.0% | 0.1% | NOISY |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float64 | 142/134 | 7.85 ms / 5.71 ms | 37.6% | 27.8% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float64 | 64/43 | 49.90 ms / 49.30 ms | 1.2% | 28.2% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | solve_ivi | Float64 | 5/5 | 7.377 s / 7.431 s | 0.7% | 0.3% | ok |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float64 | 5/5 | 2.272 s / 2.253 s | 0.9% | 0.9% | ok |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float64 | 30/27 | 49.31 ms / 41.73 ms | 18.2% | 0.0% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float64 | 25/25 | 81.63 ms / 81.14 ms | 0.6% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_ivi | Float64 | 5/5 | 9.965 s / 9.929 s | 0.4% | 0.3% | ok |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 6.728 s / 6.707 s | 0.3% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float64 | 17/16 | 65.72 ms / 70.88 ms | 7.8% | 27.1% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float64 | 20/20 | 324.33 ms / 323.33 ms | 0.3% | 20.3% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | solve_ivi | Float64 | 3/3 | 28.523 s / 28.370 s | 0.5% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float64 | 4/4 | 21.115 s / 21.085 s | 0.1% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float64 | 10/10 | 304.52 ms / 298.45 ms | 2.0% | 0.3% | ok |
| product-imdp-dense-n1000-a1-dfa4 | bellman | Float64 | 297/415 | 3.50 ms / 3.74 ms | 6.9% | 0.1% | NOISY |
| product-imdp-dense-n1000-a1-dfa4 | solve_dfa | Float64 | 7/7 | 565.47 ms / 559.87 ms | 1.0% | 0.3% | ok (clock deviating) |
| product-imdp-dense-n1000-a1-dfa4 | workspace | Float64 | 57/56 | 372.51 µs / 371.36 µs | 0.3% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | bellman | Float64 | 95/105 | 17.02 ms / 17.53 ms | 3.0% | 0.0% | ok |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | Float64 | 5/5 | 1.338 s / 1.909 s | 42.6% | 28.2% | NOISY (clock deviating) |
| product-imdp-dense-n1000-a4-dfa4 | workspace | Float64 | 345/335 | 5.65 ms / 5.66 ms | 0.1% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | Float64 | 1247/1468 | 303.98 µs / 301.77 µs | 0.7% | 28.2% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | solve_dfa | Float64 | 38/49 | 75.12 ms / 77.46 ms | 3.1% | 0.3% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | Float64 | 74/81 | 61.64 µs / 59.87 µs | 3.0% | 27.6% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | Float64 | 495/308 | 3.02 ms / 3.06 ms | 1.5% | 0.0% | ok |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | Float64 | 23/23 | 179.08 ms / 178.46 ms | 0.3% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | workspace | Float64 | 2937/3038 | 193.84 µs / 121.08 µs | 60.1% | 0.0% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | Float64 | 269/198 | 5.34 ms / 5.74 ms | 7.4% | 0.0% | NOISY |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | Float64 | 5/5 | 775.62 ms / 777.12 ms | 0.2% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | Float64 | 1166/1753 | 297.36 µs / 465.64 µs | 56.6% | 0.0% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | Float64 | 89/94 | 19.16 ms / 10.66 ms | 79.6% | 27.8% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | Float64 | 5/7 | 1.011 s / 522.31 ms | 93.5% | 27.8% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | Float64 | 363/331 | 2.18 ms / 1.29 ms | 69.0% | 0.0% | NOISY (clock deviating) |
| real-multiObj_robotIMDP | bellman | Float64 | 6317/6615 | 46.13 µs / 43.04 µs | 7.2% | 27.4% | NOISY (clock deviating) |
| real-multiObj_robotIMDP | solve_cs_stationary | Float64 | 280/258 | 5.77 ms / 5.87 ms | 1.6% | 1.3% | ok |
| real-multiObj_robotIMDP | solve_rvi | Float64 | 278/345 | 5.93 ms / 5.79 ms | 2.4% | 0.1% | ok |
| real-multiObj_robotIMDP | workspace | Float64 | 75/74 | 46.17 µs / 44.48 µs | 3.8% | 28.2% | ok (clock deviating) |

131 entries, 36 with spread > 5.0%.
spread: median 1.48%, 90th percentile 64.96%, max 142.01%

## t = 16

| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |
|---|---|---|---|---|---:|---:|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | Float64 | 15/18 | 210.66 ms / 243.82 ms | 15.7% | 0.0% | NOISY (clock deviating) |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | Float64 | 129/160 | 27.26 ms / 27.10 ms | 0.6% | 0.3% | ok (clock deviating) |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_stationary | Float64 | 5/5 | 1.160 s / 1.141 s | 1.7% | 0.3% | ok (clock deviating) |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | Float64 | 20/20 | 198.32 ms / 210.35 ms | 6.1% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a1-mccormick | bellman | Float64 | 30/30 | 85.63 ms / 84.45 ms | 1.4% | 0.0% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-mccormick | solve_rvi | Float64 | 5/5 | 7.363 s / 7.355 s | 0.1% | 0.0% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-mccormick | workspace | Float64 | 103/78 | 3.27 ms / 3.24 ms | 0.9% | 39.2% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | bellman | Float64 | 9953/9987 | 86.41 µs / 85.78 µs | 0.7% | 0.3% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float64 | 257/244 | 14.34 ms / 14.32 ms | 0.1% | 0.0% | ok (clock deviating) |
| fimdp-dense-v2-d10-a1-omax | workspace | Float64 | 120/120 | 25.48 µs / 14.86 µs | 71.5% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a4-omax | bellman | Float64 | 4053/2419 | 457.98 µs / 399.82 µs | 14.5% | 0.0% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float64 | 100/74 | 56.90 ms / 45.91 ms | 23.9% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d10-a4-omax | workspace | Float64 | 5198/5391 | 48.54 µs / 46.16 µs | 5.1% | 0.0% | NOISY (clock deviating) |
| fimdp-dense-v2-d50-a1-omax | bellman | Float64 | 71/71 | 19.64 ms / 21.43 ms | 9.2% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float64 | 5/5 | 1.702 s / 1.827 s | 7.4% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v2-d50-a1-omax | workspace | Float64 | 588/696 | 527.98 µs / 528.77 µs | 0.1% | 0.3% | ok (clock deviating) |
| fimdp-dense-v2-d50-a4-omax | bellman | Float64 | 20/21 | 86.95 ms / 85.41 ms | 1.8% | 0.0% | ok (clock deviating) |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float64 | 5/5 | 8.417 s / 8.672 s | 3.0% | 0.0% | ok (clock deviating) |
| fimdp-dense-v2-d50-a4-omax | workspace | Float64 | 179/146 | 3.11 ms / 3.08 ms | 0.8% | 0.0% | ok (clock deviating) |
| fimdp-dense-v3-d10-a1-omax | bellman | Float64 | 112/283 | 5.89 ms / 5.88 ms | 0.1% | 67.2% | ok (clock deviating) |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float64 | 6/6 | 649.30 ms / 658.33 ms | 1.4% | 0.3% | ok (clock deviating) |
| fimdp-dense-v3-d10-a1-omax | workspace | Float64 | 96/87 | 158.31 µs / 160.85 µs | 1.6% | 0.3% | ok (clock deviating) |
| fimdp-dense-v3-d10-a4-omax | bellman | Float64 | 76/58 | 29.34 ms / 27.05 ms | 8.5% | 0.3% | NOISY (clock deviating) |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float64 | 5/5 | 2.530 s / 2.654 s | 4.9% | 12.8% | ok (clock deviating) |
| fimdp-dense-v3-d10-a4-omax | workspace | Float64 | 1040/785 | 536.93 µs / 533.85 µs | 0.6% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-mccormick | bellman | Float64 | 115/99 | 18.01 ms / 18.01 ms | 0.0% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-mccormick | solve_rvi | Float64 | 5/5 | 1.884 s / 1.950 s | 3.5% | 16.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-mccormick | workspace | Float64 | 95/81 | 3.20 ms / 3.24 ms | 1.5% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | Float64 | 835/1116 | 1.88 ms / 2.04 ms | 8.7% | 0.3% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | solve_rvi | Float64 | 18/19 | 236.70 ms / 240.41 ms | 1.6% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | Float64 | 67/72 | 1.51 µs / 1.74 µs | 15.1% | 4.9% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-mccormick | bellman | Float64 | 29/30 | 80.44 ms / 79.87 ms | 0.7% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-mccormick | solve_rvi | Float64 | 5/5 | 3.491 s / 3.580 s | 2.6% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-mccormick | workspace | Float64 | 127/113 | 3.24 ms / 3.25 ms | 0.2% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | bellman | Float64 | 387/318 | 9.62 ms / 9.82 ms | 2.1% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | Float64 | 8/10 | 461.69 ms / 488.49 ms | 5.8% | 0.3% | NOISY (clock deviating) |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | Float64 | 10000/10000 | 1.83 µs / 2.03 µs | 11.0% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v2-d50-k10-a1-omax | bellman | Float64 | 783/703 | 1.83 ms / 1.83 ms | 0.2% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v2-d50-k10-a1-omax | solve_rvi | Float64 | 19/20 | 220.69 ms / 213.32 ms | 3.5% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d50-k10-a1-omax | workspace | Float64 | 108/89 | 307.49 µs / 305.16 µs | 0.8% | 5.1% | ok (clock deviating) |
| fimdp-sparse-v2-d50-k10-a4-omax | bellman | Float64 | 161/187 | 7.13 ms / 7.44 ms | 4.3% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | Float64 | 6/5 | 598.70 ms / 634.57 ms | 6.0% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v2-d50-k10-a4-omax | workspace | Float64 | 326/316 | 1.18 ms / 1.18 ms | 0.0% | 0.0% | ok (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-mccormick | bellman | Float64 | 20/20 | 339.52 ms / 334.58 ms | 1.5% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-mccormick | workspace | Float64 | 87/86 | 3.23 ms / 3.23 ms | 0.0% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | Float64 | 35/20 | 66.90 ms / 147.76 ms | 120.9% | 0.3% | NOISY (clock deviating) |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | Float64 | 58/67 | 2.19 µs / 3.04 µs | 38.8% | 0.0% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a1-omax | bellman | Float64 | 144/112 | 17.56 ms / 15.85 ms | 10.8% | 0.4% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a1-omax | solve_rvi | Float64 | 5/5 | 1.972 s / 2.010 s | 1.9% | 0.3% | ok (clock deviating) |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | Float64 | 69/76 | 1.30 ms / 1.30 ms | 0.2% | 0.1% | ok (clock deviating) |
| fimdp-sparse-v3-d20-k5-a4-omax | bellman | Float64 | 20/20 | 100.96 ms / 88.48 ms | 14.1% | 0.3% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | Float64 | 5/5 | 9.348 s / 8.828 s | 5.9% | 184.3% | NOISY (clock deviating) |
| fimdp-sparse-v3-d20-k5-a4-omax | workspace | Float64 | 145/116 | 5.02 ms / 5.01 ms | 0.1% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a1 | bellman | Float64 | 9092/5128 | 32.83 µs / 32.60 µs | 0.7% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a1 | solve_ivi | Float64 | 371/352 | 7.40 ms / 7.27 ms | 1.8% | 0.0% | ok (clock deviating) |
| imdp-dense-n100-a1 | solve_rvi | Float64 | 542/602 | 5.16 ms / 5.18 ms | 0.2% | 0.0% | ok (clock deviating) |
| imdp-dense-n100-a1 | workspace | Float64 | 114/119 | 2.63 µs / 3.50 µs | 33.4% | 0.0% | NOISY (clock deviating) |
| imdp-dense-n100-a4 | bellman | Float64 | 6047/6460 | 36.17 µs / 35.15 µs | 2.9% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_ivi | Float64 | 387/444 | 8.96 ms / 9.02 ms | 0.7% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a4 | solve_rvi | Float64 | 654/882 | 5.09 ms / 5.05 ms | 0.8% | 0.3% | ok (clock deviating) |
| imdp-dense-n100-a4 | workspace | Float64 | 10000/10000 | 5.28 µs / 5.09 µs | 3.8% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a1 | bellman | Float64 | 1817/1121 | 764.82 µs / 766.02 µs | 0.2% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a1 | solve_ivi | Float64 | 36/36 | 117.01 ms / 111.43 ms | 5.0% | 0.4% | NOISY (clock deviating) |
| imdp-dense-n1000-a1 | solve_rvi | Float64 | 60/58 | 69.79 ms / 68.85 ms | 1.4% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a1 | workspace | Float64 | 1260/1349 | 167.25 µs / 167.28 µs | 0.0% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a4 | bellman | Float64 | 525/501 | 2.89 ms / 2.94 ms | 1.6% | 0.0% | ok (clock deviating) |
| imdp-dense-n1000-a4 | solve_ivi | Float64 | 19/18 | 201.14 ms / 210.25 ms | 4.5% | 0.3% | ok (clock deviating) |
| imdp-dense-n1000-a4 | solve_rvi | Float64 | 20/19 | 207.52 ms / 250.29 ms | 20.6% | 0.0% | NOISY (clock deviating) |
| imdp-dense-n1000-a4 | workspace | Float64 | 318/345 | 5.68 ms / 5.68 ms | 0.0% | 0.0% | ok (clock deviating) |
| imdp-dense-n4000-a1 | bellman | Float64 | 180/147 | 9.73 ms / 12.72 ms | 30.7% | 0.3% | NOISY (clock deviating) |
| imdp-dense-n4000-a1 | solve_ivi | Float64 | 5/5 | 895.91 ms / 1.083 s | 20.9% | 0.0% | NOISY (clock deviating) |
| imdp-dense-n4000-a1 | solve_rvi | Float64 | 5/5 | 815.45 ms / 1.022 s | 25.3% | 0.0% | NOISY (clock deviating) |
| imdp-dense-n4000-a1 | workspace | Float64 | 90/82 | 17.14 ms / 17.18 ms | 0.2% | 0.3% | ok (clock deviating) |
| imdp-dense-n4000-a4 | bellman | Float64 | 46/44 | 41.18 ms / 43.06 ms | 4.6% | 0.0% | ok (clock deviating) |
| imdp-dense-n4000-a4 | solve_ivi | Float64 | 5/5 | 2.410 s / 2.693 s | 11.7% | 28.2% | NOISY (clock deviating) |
| imdp-dense-n4000-a4 | solve_rvi | Float64 | 5/5 | 3.628 s / 3.815 s | 5.2% | 0.3% | NOISY (clock deviating) |
| imdp-dense-n4000-a4 | workspace | Float64 | 18/18 | 71.85 ms / 73.47 ms | 2.3% | 0.3% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float64 | 1823/797 | 847.25 µs / 878.23 µs | 3.7% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | solve_ivi | Float64 | 34/30 | 125.85 ms / 121.48 ms | 3.6% | 0.3% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float64 | 43/39 | 109.89 ms / 107.95 ms | 1.8% | 28.2% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float64 | 98/98 | 1.45 ms / 1.44 ms | 0.4% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float64 | 592/806 | 3.16 ms / 2.61 ms | 20.9% | 0.3% | NOISY (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | solve_ivi | Float64 | 7/7 | 464.89 ms / 482.75 ms | 3.8% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float64 | 25/27 | 157.54 ms / 142.14 ms | 10.8% | 0.0% | NOISY (clock deviating) |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float64 | 248/283 | 2.50 ms / 2.51 ms | 0.5% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float64 | 424/409 | 4.20 ms / 4.22 ms | 0.5% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | solve_ivi | Float64 | 7/7 | 519.28 ms / 530.39 ms | 2.1% | 0.3% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float64 | 12/9 | 362.26 ms / 362.22 ms | 0.0% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float64 | 138/140 | 2.87 ms / 2.88 ms | 0.3% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float64 | 121/93 | 16.62 ms / 17.14 ms | 3.1% | 0.0% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | solve_ivi | Float64 | 5/5 | 1.562 s / 1.579 s | 1.1% | 0.3% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 1.160 s / 1.170 s | 0.9% | 0.3% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float64 | 23/24 | 41.82 ms / 41.44 ms | 0.9% | 0.3% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float64 | 197/191 | 9.69 ms / 9.46 ms | 2.4% | 0.3% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | solve_ivi | Float64 | 5/5 | 1.265 s / 1.289 s | 1.9% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float64 | 5/5 | 1.063 s / 972.30 ms | 9.4% | 0.3% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float64 | 119/93 | 6.72 ms / 6.85 ms | 2.0% | 0.3% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float64 | 37/68 | 43.56 ms / 38.90 ms | 12.0% | 0.0% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | solve_ivi | Float64 | 5/5 | 5.689 s / 6.670 s | 17.2% | 183.4% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float64 | 5/5 | 2.283 s / 1.984 s | 15.0% | 0.0% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float64 | 15/16 | 85.64 ms / 92.02 ms | 7.4% | 0.3% | NOISY (clock deviating) |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float64 | 42/38 | 49.57 ms / 49.86 ms | 0.6% | 29.9% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a1 | solve_ivi | Float64 | 5/5 | 5.982 s / 6.098 s | 1.9% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 4.228 s / 4.258 s | 0.7% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float64 | 10/10 | 128.35 ms / 127.60 ms | 0.6% | 0.4% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float64 | 20/20 | 192.83 ms / 184.98 ms | 4.2% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | solve_ivi | Float64 | 5/5 | 17.093 s / 16.958 s | 0.8% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 12.705 s / 12.608 s | 0.8% | 0.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float64 | 10/10 | 592.91 ms / 592.01 ms | 0.2% | 0.0% | ok |
| product-imdp-dense-n1000-a1-dfa4 | bellman | Float64 | 581/303 | 3.06 ms / 3.14 ms | 2.7% | 0.3% | ok (clock deviating) |
| product-imdp-dense-n1000-a1-dfa4 | solve_dfa | Float64 | 8/8 | 493.50 ms / 485.26 ms | 1.7% | 0.3% | ok (clock deviating) |
| product-imdp-dense-n1000-a1-dfa4 | workspace | Float64 | 78/74 | 174.31 µs / 201.16 µs | 15.4% | 0.3% | NOISY (clock deviating) |
| product-imdp-dense-n1000-a4-dfa4 | bellman | Float64 | 165/174 | 11.80 ms / 11.81 ms | 0.1% | 0.3% | ok (clock deviating) |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | Float64 | 5/5 | 1.240 s / 1.365 s | 10.1% | 0.0% | NOISY (clock deviating) |
| product-imdp-dense-n1000-a4-dfa4 | workspace | Float64 | 322/358 | 5.70 ms / 5.38 ms | 5.9% | 0.3% | NOISY (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | Float64 | 1858/2328 | 366.88 µs / 383.75 µs | 4.6% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | solve_dfa | Float64 | 43/41 | 99.87 ms / 100.60 ms | 0.7% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | Float64 | 73/66 | 105.39 µs / 99.93 µs | 5.5% | 0.3% | NOISY (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | Float64 | 955/472 | 2.16 ms / 1.64 ms | 32.1% | 0.2% | NOISY (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | Float64 | 29/36 | 148.59 ms / 108.74 ms | 36.6% | 0.3% | NOISY (clock deviating) |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | workspace | Float64 | 1482/722 | 239.04 µs / 240.31 µs | 0.5% | 0.3% | ok (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | Float64 | 469/534 | 4.28 ms / 3.91 ms | 9.4% | 0.3% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | Float64 | 6/6 | 741.12 ms / 646.42 ms | 14.6% | 0.3% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | Float64 | 991/722 | 589.45 µs / 585.86 µs | 0.6% | 0.0% | ok (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | Float64 | 163/170 | 10.41 ms / 11.13 ms | 6.9% | 0.0% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | Float64 | 6/6 | 577.76 ms / 614.03 ms | 6.3% | 0.0% | NOISY (clock deviating) |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | Float64 | 206/178 | 2.53 ms / 2.52 ms | 0.5% | 0.3% | ok (clock deviating) |
| real-multiObj_robotIMDP | bellman | Float64 | 10000/5667 | 61.36 µs / 60.93 µs | 0.7% | 0.6% | ok (clock deviating) |
| real-multiObj_robotIMDP | solve_cs_stationary | Float64 | 353/348 | 9.16 ms / 9.11 ms | 0.5% | 0.0% | ok (clock deviating) |
| real-multiObj_robotIMDP | solve_rvi | Float64 | 476/369 | 9.26 ms / 9.13 ms | 1.4% | 0.3% | ok (clock deviating) |
| real-multiObj_robotIMDP | workspace | Float64 | 103/97 | 52.48 µs / 87.20 µs | 66.2% | 0.3% | NOISY (clock deviating) |

131 entries, 45 with spread > 5.0%.
spread: median 2.06%, 90th percentile 17.24%, max 120.87%

## CUDA (run0-unchecked vs baseline)

| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |
|---|---|---|---|---|---:|---:|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | Float64 | 9/9 | 466.04 ms / 452.43 ms | 3.0% | 0.0% | ok INVALID |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | Float64 | 432/432 | 8.88 ms / 8.89 ms | 0.0% | 0.4% | ok INVALID |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_stationary | Float64 | 10/10 | 414.03 ms / 396.80 ms | 4.3% | 121.8% | ok (clock deviating) INVALID |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | Float64 | 149/148 | 26.98 ms / 27.06 ms | 0.3% | 0.0% | ok INVALID |
| fimdp-dense-v2-d10-a1-omax | bellman | Float32 | 4419/8611 | 43.70 µs / 43.82 µs | 0.3% | 0.2% | ok INVALID |
| fimdp-dense-v2-d10-a1-omax | bellman | Float64 | 4067/5898 | 73.07 µs / 73.26 µs | 0.3% | 2.0% | ok INVALID |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float32 | 367/364 | 7.09 ms / 7.44 ms | 5.0% | 0.2% | ok INVALID |
| fimdp-dense-v2-d10-a1-omax | solve_rvi | Float64 | 355/358 | 9.59 ms / 9.69 ms | 1.0% | 0.0% | ok INVALID |
| fimdp-dense-v2-d10-a1-omax | workspace | Float32 | 103/310 | 1.92 µs / 2.61 µs | 36.4% | 0.3% | NOISY |
| fimdp-dense-v2-d10-a1-omax | workspace | Float64 | 108/316 | 3.00 µs / 1.91 µs | 57.3% | 0.0% | NOISY |
| fimdp-dense-v2-d10-a4-omax | bellman | Float32 | 3989/9278 | 133.12 µs / 111.86 µs | 19.0% | 1.9% | NOISY INVALID |
| fimdp-dense-v2-d10-a4-omax | bellman | Float64 | 1986/3035 | 6.03 ms / 6.03 ms | 0.1% | 0.5% | ok INVALID |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float32 | 292/333 | 12.90 ms / 10.79 ms | 19.5% | 0.4% | NOISY INVALID |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | Float64 | 76/12 | 233.04 ms / 440.17 ms | 88.9% | 137.8% | NOISY (clock deviating) INVALID |
| fimdp-dense-v2-d10-a4-omax | workspace | Float32 | 10000/10000 | 1.26 µs / 1.25 µs | 0.6% | 0.9% | ok |
| fimdp-dense-v2-d10-a4-omax | workspace | Float64 | 10000/10000 | 1.26 µs / 1.25 µs | 1.0% | 0.0% | ok |
| fimdp-dense-v2-d50-a1-omax | bellman | Float32 | 1000/610 | 6.03 ms / 6.03 ms | 0.0% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a1-omax | bellman | Float64 | 304/613 | 6.12 ms / 6.13 ms | 0.0% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float32 | 8/8 | 529.47 ms / 521.81 ms | 1.5% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | Float64 | 7/7 | 535.86 ms / 534.84 ms | 0.2% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a1-omax | workspace | Float32 | 10000/10000 | 1.25 µs / 1.24 µs | 0.6% | 0.6% | ok |
| fimdp-dense-v2-d50-a1-omax | workspace | Float64 | 10000/10000 | 1.26 µs / 1.23 µs | 1.9% | 0.4% | ok |
| fimdp-dense-v2-d50-a4-omax | bellman | Float32 | 224/449 | 8.66 ms / 8.63 ms | 0.2% | 0.3% | ok INVALID |
| fimdp-dense-v2-d50-a4-omax | bellman | Float64 | 154/307 | 15.45 ms / 15.45 ms | 0.0% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float32 | 5/5 | 732.83 ms / 733.55 ms | 0.1% | 0.0% | ok INVALID |
| fimdp-dense-v2-d50-a4-omax | solve_rvi | Float64 | 5/5 | 1.277 s / 1.284 s | 0.6% | 0.4% | ok INVALID |
| fimdp-dense-v2-d50-a4-omax | workspace | Float32 | 10000/10000 | 1.25 µs / 1.25 µs | 0.4% | 4.1% | ok |
| fimdp-dense-v2-d50-a4-omax | workspace | Float64 | 10000/10000 | 1.25 µs / 1.25 µs | 0.3% | 0.0% | ok |
| fimdp-dense-v3-d10-a1-omax | bellman | Float32 | 303/599 | 6.03 ms / 6.03 ms | 0.0% | 0.4% | ok INVALID |
| fimdp-dense-v3-d10-a1-omax | bellman | Float64 | 299/601 | 6.03 ms / 6.03 ms | 0.0% | 0.6% | ok INVALID |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float32 | 7/6 | 606.17 ms / 606.52 ms | 0.1% | 0.4% | ok INVALID |
| fimdp-dense-v3-d10-a1-omax | solve_rvi | Float64 | 5/7 | 607.46 ms / 602.92 ms | 0.8% | 0.0% | ok INVALID |
| fimdp-dense-v3-d10-a1-omax | workspace | Float32 | 105/306 | 3.24 µs / 2.58 µs | 25.4% | 0.0% | NOISY |
| fimdp-dense-v3-d10-a1-omax | workspace | Float64 | 107/319 | 2.44 µs / 2.41 µs | 1.2% | 0.3% | ok |
| fimdp-dense-v3-d10-a4-omax | bellman | Float32 | 284/572 | 6.68 ms / 6.68 ms | 0.1% | 0.6% | ok INVALID |
| fimdp-dense-v3-d10-a4-omax | bellman | Float64 | 169/336 | 11.48 ms / 11.48 ms | 0.0% | 0.0% | ok INVALID |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float32 | 7/7 | 601.90 ms / 603.10 ms | 0.2% | 0.0% | ok INVALID |
| fimdp-dense-v3-d10-a4-omax | solve_rvi | Float64 | 5/5 | 1.013 s / 1.014 s | 0.1% | 0.0% | ok INVALID |
| fimdp-dense-v3-d10-a4-omax | workspace | Float32 | 10000/10000 | 1.44 µs / 1.41 µs | 1.9% | 0.0% | ok |
| fimdp-dense-v3-d10-a4-omax | workspace | Float64 | 10000/10000 | 1.43 µs / 1.41 µs | 1.3% | 0.0% | ok |
| imdp-dense-n100-a1 | bellman | Float32 | 5557/5710 | 16.79 µs / 14.22 µs | 18.1% | 121.8% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a1 | bellman | Float64 | 10/17 | 36.44 µs / 36.09 µs | 1.0% | 0.0% | ok (clock deviating) INVALID |
| imdp-dense-n100-a1 | solve_rvi | Float32 | 520/496 | 5.26 ms / 3.73 ms | 40.9% | 121.8% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a1 | solve_rvi | Float64 | 5/651 | 14.47 ms / 5.70 ms | 153.8% | 121.8% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a1 | workspace | Float32 | 98/286 | 25 ns / 25 ns | 0.0% | 0.4% | ok |
| imdp-dense-n100-a1 | workspace | Float64 | 103/301 | 25 ns / 12 ns | 108.3% | 122.5% | NOISY (clock deviating) |
| imdp-dense-n100-a4 | bellman | Float32 | 8410/10000 | 21.89 µs / 19.05 µs | 14.9% | 121.8% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a4 | bellman | Float64 | 306/5414 | 49.90 µs / 46.83 µs | 6.6% | 121.8% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a4 | solve_rvi | Float32 | 613/612 | 4.48 ms / 4.76 ms | 6.3% | 121.7% | NOISY (clock deviating) INVALID |
| imdp-dense-n100-a4 | solve_rvi | Float64 | 591/609 | 5.28 ms / 5.39 ms | 2.0% | 0.2% | ok INVALID |
| imdp-dense-n100-a4 | workspace | Float32 | 10000/10000 | 5 ns / 5 ns | 0.0% | 0.0% | ok |
| imdp-dense-n100-a4 | workspace | Float64 | 10000/10000 | 3 ns / 4 ns | 25.1% | 3.9% | NOISY |
| imdp-dense-n1000-a1 | bellman | Float32 | 5488/5501 | 59.91 µs / 60.38 µs | 0.8% | 0.3% | ok INVALID |
| imdp-dense-n1000-a1 | bellman | Float64 | 1944/311 | 6.02 ms / 5.94 ms | 1.2% | 122.7% | ok (clock deviating) INVALID |
| imdp-dense-n1000-a1 | solve_rvi | Float32 | 271/298 | 7.16 ms / 7.27 ms | 1.5% | 0.0% | ok INVALID |
| imdp-dense-n1000-a1 | solve_rvi | Float64 | 38/39 | 472.86 ms / 469.98 ms | 0.6% | 0.3% | ok INVALID |
| imdp-dense-n1000-a1 | workspace | Float32 | 10000/10000 | 5 ns / 5 ns | 0.0% | 132.3% | ok (clock deviating) |
| imdp-dense-n1000-a1 | workspace | Float64 | 10000/10000 | 5 ns / 4 ns | 24.7% | 0.0% | NOISY |
| imdp-dense-n1000-a4 | bellman | Float32 | 320/562 | 6.02 ms / 6.02 ms | 0.0% | 125.2% | ok (clock deviating) INVALID |
| imdp-dense-n1000-a4 | bellman | Float64 | 311/311 | 6.02 ms / 6.02 ms | 0.1% | 0.4% | ok INVALID |
| imdp-dense-n1000-a4 | solve_rvi | Float32 | 12/10 | 462.39 ms / 457.24 ms | 1.1% | 0.2% | ok INVALID |
| imdp-dense-n1000-a4 | solve_rvi | Float64 | 9/9 | 456.30 ms / 457.33 ms | 0.2% | 0.0% | ok INVALID |
| imdp-dense-n1000-a4 | workspace | Float32 | 10000/10000 | 5 ns / 4 ns | 14.7% | 0.0% | NOISY |
| imdp-dense-n1000-a4 | workspace | Float64 | 10000/10000 | 4 ns / 4 ns | 2.9% | 0.3% | ok |
| imdp-dense-n4000-a1 | bellman | Float32 | 310/304 | 6.02 ms / 5.93 ms | 1.6% | 122.5% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a1 | bellman | Float64 | 260/309 | 6.02 ms / 5.92 ms | 1.7% | 121.8% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a1 | solve_rvi | Float32 | 8/8 | 489.05 ms / 484.93 ms | 0.8% | 0.0% | ok INVALID |
| imdp-dense-n4000-a1 | solve_rvi | Float64 | 8/9 | 479.80 ms / 472.95 ms | 1.4% | 121.8% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a1 | workspace | Float32 | 10000/10000 | 5 ns / 2 ns | 148.1% | 108.7% | NOISY (clock deviating) |
| imdp-dense-n4000-a1 | workspace | Float64 | 10000/10000 | 4 ns / 2 ns | 110.0% | 122.5% | NOISY (clock deviating) |
| imdp-dense-n4000-a4 | bellman | Float32 | 237/237 | 7.68 ms / 7.70 ms | 0.2% | 109.8% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a4 | bellman | Float64 | 164/161 | 11.73 ms / 12.11 ms | 3.2% | 121.8% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a4 | solve_rvi | Float32 | 7/7 | 569.24 ms / 564.76 ms | 0.8% | 0.0% | ok INVALID |
| imdp-dense-n4000-a4 | solve_rvi | Float64 | 5/5 | 957.68 ms / 965.98 ms | 0.9% | 117.7% | ok (clock deviating) INVALID |
| imdp-dense-n4000-a4 | workspace | Float32 | 10000/10000 | 5 ns / 2 ns | 112.5% | 122.3% | NOISY (clock deviating) |
| imdp-dense-n4000-a4 | workspace | Float64 | 10000/10000 | 4 ns / 2 ns | 146.1% | 122.2% | NOISY (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float32 | 4264/4188 | 90.50 µs / 90.94 µs | 0.5% | 109.4% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz10-a1 | bellman | Float64 | 307/303 | 6.03 ms / 6.04 ms | 0.0% | 0.1% | ok INVALID |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float32 | 305/302 | 11.59 ms / 11.80 ms | 1.8% | 0.1% | ok INVALID |
| imdp-sparse-n10000-nnz10-a1 | solve_rvi | Float64 | 7/7 | 539.38 ms / 552.86 ms | 2.5% | 0.0% | ok INVALID |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float32 | 93/269 | 28.40 µs / 27.34 µs | 3.9% | 121.8% | ok (clock deviating) |
| imdp-sparse-n10000-nnz10-a1 | workspace | Float64 | 135/406 | 28.32 µs / 25.78 µs | 9.8% | 0.0% | NOISY |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float32 | 683/313 | 6.03 ms / 6.02 ms | 0.0% | 0.4% | ok INVALID |
| imdp-sparse-n10000-nnz10-a4 | bellman | Float64 | 308/309 | 6.03 ms / 6.04 ms | 0.2% | 121.8% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float32 | 15/18 | 263.12 ms / 264.45 ms | 0.5% | 121.8% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | Float64 | 15/15 | 268.00 ms / 268.48 ms | 0.2% | 0.2% | ok INVALID |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float32 | 3648/8218 | 26.97 µs / 27.49 µs | 1.9% | 0.0% | ok |
| imdp-sparse-n10000-nnz10-a4 | workspace | Float64 | 4839/10000 | 26.95 µs / 27.65 µs | 2.6% | 121.8% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float32 | 306/312 | 6.03 ms / 6.01 ms | 0.3% | 0.0% | ok INVALID |
| imdp-sparse-n10000-nnz100-a1 | bellman | Float64 | 1649/313 | 6.03 ms / 5.94 ms | 1.5% | 122.5% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float32 | 8/9 | 502.86 ms / 495.84 ms | 1.4% | 0.2% | ok INVALID |
| imdp-sparse-n10000-nnz100-a1 | solve_rvi | Float64 | 8/10 | 507.58 ms / 501.56 ms | 1.2% | 121.8% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float32 | 2948/8740 | 22.89 µs / 18.22 µs | 25.6% | 2.9% | NOISY |
| imdp-sparse-n10000-nnz100-a1 | workspace | Float64 | 2956/4302 | 21.95 µs / 22.88 µs | 4.2% | 167.6% | ok (clock deviating) |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float32 | 307/308 | 6.03 ms / 6.02 ms | 0.0% | 0.2% | ok INVALID |
| imdp-sparse-n10000-nnz100-a4 | bellman | Float64 | 308/303 | 6.03 ms / 6.03 ms | 0.0% | 121.8% | ok (clock deviating) INVALID |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float32 | 10/10 | 399.70 ms / 401.47 ms | 0.4% | 0.4% | ok INVALID |
| imdp-sparse-n10000-nnz100-a4 | solve_rvi | Float64 | 10/10 | 400.43 ms / 396.84 ms | 0.9% | 0.6% | ok INVALID |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float32 | 3531/9851 | 27.33 µs / 27.62 µs | 1.0% | 0.0% | ok |
| imdp-sparse-n10000-nnz100-a4 | workspace | Float64 | 2632/9624 | 27.16 µs / 27.36 µs | 0.7% | 1.4% | ok |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float32 | 308/310 | 6.03 ms / 6.03 ms | 0.0% | 0.3% | ok INVALID |
| imdp-sparse-n100000-nnz10-a1 | bellman | Float64 | 310/308 | 6.03 ms / 6.03 ms | 0.0% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float32 | 7/7 | 563.05 ms / 553.97 ms | 1.6% | 122.6% | ok (clock deviating) INVALID |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | Float64 | 7/7 | 566.43 ms / 565.46 ms | 0.2% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float32 | 3225/8621 | 37.37 µs / 38.01 µs | 1.7% | 0.0% | ok |
| imdp-sparse-n100000-nnz10-a1 | workspace | Float64 | 3336/8301 | 37.39 µs / 38.03 µs | 1.7% | 0.4% | ok |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float32 | 309/312 | 6.03 ms / 6.03 ms | 0.0% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz10-a4 | bellman | Float64 | 162/161 | 12.08 ms / 12.08 ms | 0.0% | 121.9% | ok (clock deviating) INVALID |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float32 | 15/15 | 270.64 ms / 270.00 ms | 0.2% | 0.5% | ok INVALID |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | Float64 | 7/7 | 551.79 ms / 550.78 ms | 0.2% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float32 | 2143/8487 | 86.49 µs / 86.45 µs | 0.0% | 26.0% | ok (clock deviating) |
| imdp-sparse-n100000-nnz10-a4 | workspace | Float64 | 2525/2881 | 85.45 µs / 5.97 ms | 6886.9% | 0.7% | NOISY |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float32 | 311/322 | 6.13 ms / 6.15 ms | 0.3% | 0.2% | ok INVALID |
| imdp-sparse-n100000-nnz100-a1 | bellman | Float64 | 206/151 | 9.33 ms / 9.36 ms | 0.3% | 1.2% | ok INVALID |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float32 | 7/8 | 516.16 ms / 522.43 ms | 1.2% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz100-a1 | solve_rvi | Float64 | 5/5 | 789.24 ms / 787.87 ms | 0.2% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float32 | 2914/10000 | 37.61 µs / 38.03 µs | 1.1% | 0.3% | ok |
| imdp-sparse-n100000-nnz100-a1 | workspace | Float64 | 2741/10000 | 37.32 µs / 38.24 µs | 2.5% | 0.2% | ok |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float32 | 162/163 | 16.21 ms / 16.27 ms | 0.4% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz100-a4 | bellman | Float64 | 76/85 | 27.14 ms / 27.29 ms | 0.5% | 0.9% | ok INVALID |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float32 | 5/5 | 1.126 s / 1.109 s | 1.5% | 1.6% | ok INVALID |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi | Float64 | 5/5 | 1.876 s / 1.884 s | 0.4% | 0.0% | ok INVALID |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float32 | 2391/10000 | 86.05 µs / 86.85 µs | 0.9% | 0.0% | ok |
| imdp-sparse-n100000-nnz100-a4 | workspace | Float64 | 2460/8770 | 86.04 µs / 86.80 µs | 0.9% | 4.0% | ok |
| real-multiObj_robotIMDP | bellman | Float64 | 4747/9296 | 44.03 µs / 43.94 µs | 0.2% | 131.8% | ok (clock deviating) INVALID |
| real-multiObj_robotIMDP | solve_cs_stationary | Float64 | 240/416 | 12.52 ms / 4.83 ms | 158.9% | 122.3% | NOISY (clock deviating) INVALID |
| real-multiObj_robotIMDP | solve_rvi | Float64 | 259/447 | 12.54 ms / 6.37 ms | 96.9% | 0.0% | NOISY INVALID |
| real-multiObj_robotIMDP | workspace | Float64 | 86/257 | 30.86 µs / 27.40 µs | 12.6% | 0.5% | NOISY |

128 entries, 26 with spread > 5.0%.
spread: median 0.92%, 90th percentile 28.83%, max 6886.89%

## Causes of the > 5% entries and re-measurement (sub-phase 0d)

Cause codes: **CLK** clock probes of the two runs differ by > 5% (firmware clock cap ≈ 2.3 GHz vs ≈ 5.1 GHz,
REPORT.md § 1.1; at t = 16 also the 436 → 557 µs all-core probe); **WS** `workspace` entry with equal clocks
(fresh-array allocation, first-touch page faults, GC state differ between processes); **THR** threaded entry
(t ≥ 4) with equal clocks (static chunking on hybrid P/E/LP-E cores: the slowest core decides, and it differs
between processes); **ST** single-threaded compute entry with equal clocks (process-level variation).

Before: baseline vs rerun1 (budgets bellman 2 s/≥20, workspace 1 s/≥10, solves ×1). After: `results/noisefix/`
run a vs run b, separate processes ≈ 40 min apart (budgets bellman 4 s/≥40, workspace 3 s/≥50, solves
`--budget-scale 2`). Clock probe = `clock_probe_before_ns` of the entry in each run.

### t = 1

| case | entry | spread before | cause before | clock probes before (µs) | spread after | clock probes after (µs) | cause after |
|---|---|---:|---|---|---:|---|---|
| fimdp-dense-v3-d10-a1-omax | workspace | 9.4% | WS | 436 / 436 | 1.2% | 437 / 209 | fixed |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 6.9% | WS | 438 / 439 | 32.6% | 200 / 197 | WS |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 8.1% | ST | 445 / 436 | 13.7% | 557 / 197 | CLK |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 27.1% | WS | 436 / 436 | 11.3% | 197 / 197 | WS |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | 22.1% | WS | 436 / 438 | 81.1% | 197 / 197 | WS |
| imdp-dense-n100-a1 | workspace | 32.3% | WS | 436 / 437 | 41.1% | 197 / 197 | WS |
| imdp-dense-n1000-a4 | workspace | 36.5% | WS | 436 / 437 | 0.9% | 436 / 197 | fixed |
| imdp-sparse-n10000-nnz10-a1 | workspace | 31.8% | WS | 436 / 438 | 9.1% | 197 / 197 | WS |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | 12.8% | WS | 436 / 436 | 7.0% | 203 / 197 | WS |

9 entries > 5% before; causes {'ST': 1, 'WS': 8}. After re-measurement: 2 ≤ 5%, 7 still > 5% ({'CLK': 1, 'WS': 6}). Spread after: median 11.3%.

### t = 4

| case | entry | spread before | cause before | clock probes before (µs) | spread after | clock probes after (µs) | cause after |
|---|---|---:|---|---|---:|---|---|
| fimdp-dense-v2-d10-a1-omax | workspace | 6.1% | WS | 436 / 437 | 70.8% | 436 / 209 | CLK |
| fimdp-dense-v3-d10-a1-omax | bellman | 10.4% | THR | 436 / 437 | 72.0% | 559 / 199 | CLK |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 42.7% | CLK | 557 / 436 | 44.8% | 436 / 197 | CLK |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 5.9% | WS | 436 / 437 | 96.1% | 436 / 197 | CLK |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | 20.5% | WS | 437 / 436 | 38.0% | 436 / 197 | CLK |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 6.4% | THR | 436 / 436 | 0.4% | 436 / 197 | fixed |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 17.0% | WS | 437 / 436 | 9.4% | 437 / 197 | CLK |
| imdp-dense-n100-a1 | bellman | 118.8% | CLK | 436 / 559 | 5.7% | 436 / 197 | CLK |
| imdp-dense-n100-a1 | solve_ivi | 14.7% | THR | 436 / 436 | 0.2% | 437 / 197 | fixed |
| imdp-dense-n100-a1 | workspace | 12.2% | WS | 436 / 437 | 12.8% | 436 / 197 | CLK |
| imdp-dense-n1000-a1 | workspace | 118.9% | CLK | 436 / 557 | 1.4% | 436 / 197 | fixed |
| imdp-sparse-n10000-nnz10-a4 | bellman | 18.4% | THR | 436 / 436 | 13.9% | 559 / 197 | CLK |
| imdp-sparse-n10000-nnz100-a1 | bellman | 22.4% | THR | 437 / 436 | 90.5% | 559 / 197 | CLK |
| imdp-sparse-n10000-nnz100-a1 | solve_ivi | 5.1% | THR | 436 / 436 | 83.7% | 436 / 197 | CLK |
| imdp-sparse-n100000-nnz100-a1 | workspace | 7.7% | WS | 437 / 436 | 34.0% | 436 / 197 | CLK |
| imdp-sparse-n100000-nnz100-a4 | bellman | 8.7% | THR | 436 / 436 | 16.9% | 436 / 197 | CLK |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | 7.1% | THR | 436 / 438 | 18.4% | 436 / 197 | CLK |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | 13.4% | WS | 437 / 436 | 2.2% | 436 / 436 | fixed |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | 54.9% | CLK | 436 / 557 | 2.1% | 436 / 197 | fixed |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 8.4% | THR | 436 / 437 | 65.2% | 436 / 197 | CLK |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 5.9% | THR | 436 / 436 | 58.4% | 436 / 197 | CLK |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | 10.4% | THR | 436 / 436 | 6.1% | 436 / 197 | CLK |

22 entries > 5% before; causes {'CLK': 4, 'THR': 11, 'WS': 7}. After re-measurement: 5 ≤ 5%, 17 still > 5% ({'CLK': 17}). Spread after: median 17.7%.

### t = 8

| case | entry | spread before | cause before | clock probes before (µs) | spread after | clock probes after (µs) | cause after |
|---|---|---:|---|---|---:|---|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | 7.2% | THR | 557 / 559 | 52.7% | 436 / 557 | CLK |
| cs-imdp-dense-n1000-a4 | solve_cs_timevarying | 35.2% | THR | 436 / 436 | 2.5% | 559 / 436 | fixed |
| fimdp-dense-v2-d10-a1-mccormick | workspace | 114.7% | CLK | 557 / 437 | 5.4% | 559 / 559 | WS |
| fimdp-dense-v2-d10-a1-omax | bellman | 123.7% | CLK | 559 / 437 | 14.5% | 436 / 197 | CLK |
| fimdp-dense-v2-d10-a1-omax | workspace | 36.8% | CLK | 559 / 437 | 0.4% | 559 / 557 | fixed |
| fimdp-dense-v2-d10-a4-omax | workspace | 12.8% | WS | 557 / 557 | 5.4% | 209 / 560 | CLK |
| fimdp-dense-v2-d50-a4-omax | workspace | 106.4% | WS | 559 / 559 | 1.1% | 559 / 197 | fixed |
| fimdp-dense-v3-d10-a1-omax | bellman | 142.0% | CLK | 436 / 557 | 139.7% | 436 / 209 | CLK |
| fimdp-dense-v3-d10-a1-omax | workspace | 9.3% | WS | 436 / 436 | 3.3% | 559 / 501 | fixed |
| fimdp-dense-v3-d10-a4-omax | workspace | 43.6% | WS | 557 / 557 | 1.6% | 561 / 559 | fixed |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 14.0% | THR | 436 / 436 | 6.5% | 438 / 197 | CLK |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 16.6% | WS | 557 / 557 | 1.9% | 567 / 557 | fixed |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | 93.0% | CLK | 557 / 436 | 1.5% | 197 / 559 | fixed |
| fimdp-sparse-v2-d50-k10-a1-omax | bellman | 95.0% | CLK | 437 / 557 | 96.3% | 436 / 559 | CLK |
| fimdp-sparse-v2-d50-k10-a1-omax | workspace | 41.5% | CLK | 437 / 559 | 0.6% | 559 / 197 | fixed |
| fimdp-sparse-v2-d50-k10-a4-omax | workspace | 65.0% | WS | 557 / 557 | 6.0% | 560 / 197 | CLK |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 16.4% | CLK | 437 / 557 | 10.5% | 436 / 197 | CLK |
| fimdp-sparse-v3-d20-k5-a1-omax | workspace | 75.9% | CLK | 557 / 441 | 81.8% | 436 / 197 | CLK |
| imdp-dense-n100-a1 | bellman | 85.2% | THR | 436 / 436 | 14.1% | 436 / 209 | CLK |
| imdp-dense-n100-a4 | workspace | 5.4% | WS | 559 / 557 | 64.8% | 540 / 559 | WS |
| imdp-dense-n1000-a1 | bellman | 5.1% | CLK | 437 / 559 | 7.7% | 436 / 197 | CLK |
| imdp-sparse-n10000-nnz10-a4 | bellman | 118.0% | THR | 436 / 436 | 5.2% | 436 / 197 | CLK |
| imdp-sparse-n10000-nnz100-a4 | workspace | 7.0% | WS | 557 / 559 | 4.6% | 557 / 563 | fixed |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | 124.0% | THR | 436 / 436 | 2.4% | 209 / 559 | fixed |
| imdp-sparse-n100000-nnz10-a1 | workspace | 37.6% | CLK | 436 / 557 | 7.6% | 197 / 558 | CLK |
| imdp-sparse-n100000-nnz10-a4 | workspace | 18.2% | WS | 557 / 557 | 31.0% | 559 / 560 | WS |
| imdp-sparse-n100000-nnz100-a1 | workspace | 7.8% | CLK | 438 / 557 | 0.3% | 557 / 559 | fixed |
| product-imdp-dense-n1000-a1-dfa4 | bellman | 6.9% | THR | 437 / 438 | 3.2% | 436 / 197 | fixed |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | 42.6% | CLK | 559 / 436 | 0.4% | 209 / 562 | fixed |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | workspace | 60.1% | WS | 559 / 559 | 0.4% | 202 / 197 | fixed |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 7.4% | THR | 436 / 436 | 21.0% | 436 / 197 | CLK |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | 56.6% | WS | 559 / 559 | 43.9% | 456 / 559 | CLK |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 79.6% | CLK | 436 / 557 | 5.1% | 559 / 197 | CLK |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | 93.5% | CLK | 436 / 557 | 1.3% | 197 / 559 | fixed |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | 69.0% | WS | 559 / 559 | 71.0% | 299 / 557 | CLK |
| real-multiObj_robotIMDP | bellman | 7.2% | CLK | 437 / 557 | 14.6% | 557 / 199 | CLK |

36 entries > 5% before; causes {'CLK': 16, 'THR': 8, 'WS': 12}. After re-measurement: 15 ≤ 5%, 21 still > 5% ({'CLK': 18, 'WS': 3}). Spread after: median 5.4%.

### t = 16

| case | entry | spread before | cause before | clock probes before (µs) | spread after | clock probes after (µs) | cause after |
|---|---|---:|---|---|---:|---|---|
| cs-imdp-dense-n1000-a4 | solve_cs_stationary | 15.7% | THR | 557 / 557 | 2.4% | 566 / 197 | fixed |
| cs-imdp-sparse-n10000-nnz100-a4 | solve_cs_timevarying | 6.1% | THR | 559 / 557 | 4.9% | 557 / 559 | fixed |
| fimdp-dense-v2-d10-a1-omax | workspace | 71.5% | WS | 559 / 557 | 0.2% | 197 / 558 | fixed |
| fimdp-dense-v2-d10-a4-omax | bellman | 14.5% | THR | 557 / 557 | 0.9% | 197 / 197 | fixed |
| fimdp-dense-v2-d10-a4-omax | solve_rvi | 23.9% | THR | 557 / 559 | 2.4% | 197 / 560 | fixed |
| fimdp-dense-v2-d10-a4-omax | workspace | 5.1% | WS | 557 / 557 | 0.8% | 557 / 559 | fixed |
| fimdp-dense-v2-d50-a1-omax | bellman | 9.2% | THR | 557 / 559 | 7.8% | 197 / 197 | THR |
| fimdp-dense-v2-d50-a1-omax | solve_rvi | 7.4% | THR | 559 / 557 | 11.0% | 197 / 560 | CLK |
| fimdp-dense-v3-d10-a4-omax | bellman | 8.5% | THR | 557 / 559 | 4.5% | 197 / 197 | fixed |
| fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 8.7% | THR | 557 / 559 | 6.6% | 201 / 197 | THR |
| fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 15.1% | WS | 532 / 559 | 20.2% | 197 / 559 | CLK |
| fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | 5.8% | THR | 559 / 557 | 10.7% | 197 / 559 | CLK |
| fimdp-sparse-v2-d10-k4-a4-vertex | workspace | 11.0% | WS | 557 / 557 | 12.1% | 557 / 209 | CLK |
| fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | 6.0% | THR | 557 / 557 | 5.6% | 197 / 559 | CLK |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 120.9% | THR | 559 / 557 | 41.2% | 209 / 197 | CLK |
| fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 38.8% | WS | 557 / 557 | 15.4% | 563 / 557 | WS |
| fimdp-sparse-v3-d20-k5-a1-omax | bellman | 10.8% | THR | 557 / 559 | 6.5% | 197 / 197 | THR |
| fimdp-sparse-v3-d20-k5-a4-omax | bellman | 14.1% | THR | 559 / 557 | 5.1% | 197 / 198 | THR |
| fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | 5.9% | CLK | 197 / 559 | 23.7% | 197 / 197 | THR |
| imdp-dense-n100-a1 | workspace | 33.4% | WS | 559 / 559 | 29.1% | 436 / 559 | CLK |
| imdp-dense-n1000-a1 | solve_ivi | 5.0% | THR | 559 / 557 | 1.6% | 197 / 209 | fixed |
| imdp-dense-n1000-a4 | solve_rvi | 20.6% | THR | 559 / 559 | 5.5% | 197 / 557 | CLK |
| imdp-dense-n4000-a1 | bellman | 30.7% | THR | 559 / 557 | 2.4% | 197 / 209 | fixed |
| imdp-dense-n4000-a1 | solve_ivi | 20.9% | THR | 559 / 559 | 27.3% | 197 / 436 | CLK |
| imdp-dense-n4000-a1 | solve_rvi | 25.3% | THR | 559 / 559 | 8.1% | 197 / 559 | CLK |
| imdp-dense-n4000-a4 | solve_ivi | 11.7% | CLK | 436 / 559 | 5.2% | 197 / 436 | CLK |
| imdp-dense-n4000-a4 | solve_rvi | 5.2% | THR | 559 / 557 | 0.2% | 348 / 559 | fixed |
| imdp-sparse-n10000-nnz10-a4 | bellman | 20.9% | THR | 557 / 559 | 13.7% | 197 / 197 | THR |
| imdp-sparse-n10000-nnz10-a4 | solve_rvi | 10.8% | THR | 559 / 559 | 2.3% | 197 / 559 | fixed |
| imdp-sparse-n100000-nnz10-a1 | solve_rvi | 9.4% | THR | 557 / 559 | 3.9% | 197 / 560 | fixed |
| imdp-sparse-n100000-nnz10-a4 | bellman | 12.0% | THR | 557 / 557 | 19.5% | 197 / 197 | THR |
| imdp-sparse-n100000-nnz10-a4 | solve_ivi | 17.2% | CLK | 557 / 197 | 14.7% | 203 / 197 | THR |
| imdp-sparse-n100000-nnz10-a4 | solve_rvi | 15.0% | THR | 559 / 559 | 0.6% | 197 / 559 | fixed |
| imdp-sparse-n100000-nnz10-a4 | workspace | 7.4% | WS | 557 / 559 | 7.2% | 559 / 559 | WS |
| product-imdp-dense-n1000-a1-dfa4 | workspace | 15.4% | WS | 557 / 559 | 7.4% | 197 / 559 | CLK |
| product-imdp-dense-n1000-a4-dfa4 | solve_dfa | 10.1% | THR | 559 / 559 | 1.9% | 560 / 559 | fixed |
| product-imdp-dense-n1000-a4-dfa4 | workspace | 5.9% | WS | 557 / 559 | 0.4% | 559 / 560 | fixed |
| product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | 5.5% | WS | 559 / 557 | 1.8% | 561 / 559 | fixed |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | 32.1% | THR | 559 / 558 | 13.4% | 197 / 197 | THR |
| product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | 36.6% | THR | 559 / 557 | 7.7% | 557 / 557 | THR |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 9.4% | THR | 559 / 557 | 59.0% | 197 / 197 | THR |
| product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | 14.6% | THR | 557 / 559 | 14.1% | 560 / 559 | THR |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 6.9% | THR | 559 / 559 | 4.2% | 197 / 197 | fixed |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | 6.3% | THR | 557 / 557 | 3.8% | 336 / 559 | fixed |
| real-multiObj_robotIMDP | workspace | 66.2% | WS | 559 / 557 | 15.9% | 209 / 559 | CLK |

45 entries > 5% before; causes {'CLK': 3, 'THR': 31, 'WS': 11}. After re-measurement: 18 ≤ 5%, 27 still > 5% ({'CLK': 13, 'THR': 12, 'WS': 2}). Spread after: median 6.5%.

