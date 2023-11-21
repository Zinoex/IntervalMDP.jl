# Usage

The general usage of this package can be described as 3 steps
1. Construct interval Markov process (IMC or IMDP)
2. Choose specification (reachability or reach-avoid)
3. Call `value_iteration` or `satisfaction_prob`.

First, we construct a system. We can either construct an interval Markov chain or an interval Markov decision process.



## Sparse matrices
- Why sparsity?
- CSC format
- How to efficiently construct it
- Why CSC format

## CUDA
- Pacakges that are required for CUDA to work
- Transfer matrices vs transfer model