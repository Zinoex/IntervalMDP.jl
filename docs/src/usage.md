# Usage

1. Construct system (transition probability, initial state)
2. Construct specification (temporal logic formula or reachability/avoidance)
3. Either call `interval_value_iteration` or `satisfaction_prob` depending on whether access to the value function is required or not.


## Sparse matrices
- Why sparsity?
- CSC format
- How to efficiently construct it
- Why CSC format

## CUDA
- Pacakges that are required for CUDA to work
- Transfer matrices vs transfer model