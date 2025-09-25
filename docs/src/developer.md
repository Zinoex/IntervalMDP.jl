# Developer documentation

## Bellman algorithms
### [O-maximization](@id dev-docs-omax)
To optimize the procedure, we abstract the O-maximization algorithm into the sorting phase and the O-maximization phase: 
```julia
function min_value(V, system, source, action)
    # Sort values of `V` in ascending order
    order = sortstates(V)
    v = o_maximize(system, source, action, order)
    return v
end
```
Notice the the order is shared for all source-action pairs, and thus, we can pre-compute it once per Bellman update. We however only do so for dense transition ambiguity sets, as in the sparse case, it is often faster to sort repeatedly, but only for the support. I.e.,
```julia
function sortstates(V, system, source, action)  # I.e. sort per source-action pair
    supp = support(system, source, action)
    order = sortperm(@view(V[supp])) # Sort only for the support
    return supp[order]  # Return sorted indices in original indexing
end
```

#### GPU acceleration
The sorting and O-maximization phases can be parallelized on the GPU to leverage the massive parallelism. The following assumes that the reader is familiar with the CUDA programming model; see [CUDA Programming Model](@ref) for a brief introduction. The specific execution plan depends on the storage type and size of model; please refer to the source code for specifics.

##### Sorting
Sorting in parallel on the GPU is a well-studied problem, and there are many algorithms for doing so. We choose to use bitonic sorting, which is a sorting network that is easily parallelized and implementable on a GPU. The idea is to merge bitonic subsets, i.e. sets with first increasing then decreasing subsets of equal size, of increasingly larger sizes and perform minor rounds of swaps to maintain the bitonic property. The figure below shows 3 major rounds to sort a set of 8 elements (each line represents an element, each arrow is a comparison pointing towards the larger element). The latency[^1] of the sorting network is ``O((\lg n)^2)``, and thus it scales well to larger number of elements. See [Wikipedia](https://en.wikipedia.org/wiki/Bitonic_sorter) for more details.

![](assets/bitonic_sorting.svg)


##### O-maximization phase
In order to parallelize the O-maximization phase, observe that O-maximization implicity implements a cumulative sum according to the ordering over gaps and this is the only dependency between the states. Hence, if we can parallelize this cumulative sum, then we can parallelize the O-maximization phase.
Luckily, there is a well-studied algorithm for computing the cumulative sum in parallel: tree reduction for prefix scan. The idea is best explained with figure below.

![](assets/tree_reduction_prefix_scan.svg)

Here, we recursively compute the cumulative sum of larger and larger subsets of the array. The latency is ``O(\lg n)``, and thus very efficient. See [Wikipedia](https://en.wikipedia.org/wiki/Prefix_sum) for more details. Putting it all together, we get the following (pseudo-code) algorithm for O-maximization:
```julia
function o_maximize(system, source, action, order)
    p = lower_bounds(system, source, action)
    rem = 1 - sum(p)
    gap = upper_bounds(system, source, action) - p

    # Ordered cumulative sum of gaps via tree reduction
    cumgap = cumulative_sum(gap[order])

    @parallelize for (i, o) in enumerate(order)
        rem_state = max(rem - cumgap[i] + gap[o], 0)
        if gap[o] < rem_state
            p[o] += gap[o]
        else
            p[o] += rem_state
            break
        end
    end

    return p
end
```
When implementing the algorithm above in CUDA, it is possible to use warp shuffles to very efficiently perform tree reductions of up to 32 elements. For larger sets, shared memory to store the intermediate results, which is much faster than global memory. See [CUDA Programming Model](@ref) for more details on why these choices are important.

### [Vertex enumeration](@id dev-docs-vertex-enumeration)
First, we concern ourselves with enumerating the vertices of a single marginal. The key observation for an efficient algorithm is that, while each vertex corresponds to a unique ordering of the states, many orderings yield the same vertex. Thus, we need an algorithm that generates each vertex exactly once without generating all orderings explicitly. To this end, we rely on a backtracking algorithm where state values are added to a list of a "maximizing" state values, and backtrack once a vertex is found, i.e. `sum(p) == 1` and the remaining state values are assigned a lower bound.

For the product of marginals, we simply apply ``Iterators.product`` to get an iterator over all combinations of vertices.

### [Recursive McCormick envelopes](@id dev-docs-mccormick)
The recursive McCormick envelopes for polytoptic fRMDPs are described in [schnitzer2025efficient](@cite), with the addition that we add the marginal constraints to the linear program and the constraint that each relaxation `q` in the recursion is a valid probability distribution, i.e. `sum(q) == 1`. 

Another consideration is whether to recursively relax as a sequence of marginals or as a binary tree. In [schnitzer2025efficient](@cite), the recursive relaxation is done as a sequence. However, the tree structure requires significantly fewer auxiliary variables and thus both memory and time. A formal argument of the resulting minimum value between the two relaxation structures is missing, but empirically, they yield the same results.


## CUDA Programming Model
We here give a brief introduction to the CUDA programming model to understand to algorithmic choices. For a more in-depth introduction, see the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html). The CUDA framework is Single-Instruction Multiple-Thread (SIMT) parallel execution platform and Application Programming Interface. This is in contrast to Single-Instruction Multiple-Data where all data must be processed homogeneously without control flow. SIMT makes CUDA more flexible for heterogeneous processing and control flow. The smallest execution unit in CUDA is a thread, which is a sequential processing of instructions. A thread is uniquely identified by its thread index, which allows indexing into the global data for parallel processing. A group of 32 threads[^2] is called a warp, which will be executed _mostly_ synchronously on a streaming multiprocessor. If control flow makes threads in a wrap diverge, instructions may need to be decoded twice and executed in two separate cycles. Due to this synchronous behavior, data can be shared in registers between threads in a warp for maximum performance. A collection of (up to) 1024 threads is called a block, and this is the largest aggregation that can be synchronized. Furthermore, threads in a block share the appropriately named shared memory. This is memory that is stored locally on the streaming multiprocessor for fast access. Note that shared memory is unintuitively faster than local memory (not to be confused with registers) due to local memory being allocated in device memory. Finally, a collection of (up to) 65536 blocks is called the grid of a kernel, which is the set of instructions to be executed. The grid is singular as only a single ever exists per launched kernel. Hence, if more blocks are necessary to process the amount of data, then a grid-strided loop or multiple kernels are necessary. 

![](assets/cuda_programming_model.svg)


[^1]: Note that when assessing parallel algorithms, the asymptotic performance is measured by the latency, which is the delay in the number of parallel operations, before the result is available. This is in contrast to traditional algorithms, which are assessed by the total number of operations.

[^2]: with consecutive thread indices aligned to a multiple of 32.