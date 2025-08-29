"""
    IntervalAmbiguitySets{R, MR <: AbstractMatrix{R}, N, M, I}

A matrix pair to represent the lower and upper bound transition probabilities from all source/action pairs to all target states.
The matrices can be `Matrix{R}` or `SparseMatrixCSC{R}`, or their CUDA equivalents. For memory efficiency, it is recommended to use sparse matrices.

The columns represent the source and the rows represent the target, as if the probability matrix was a linear transformation.
Mathematically, let ``P`` be the probability matrix. Then ``P_{ij}`` represents the probability of transitioning from state ``j`` (or with state/action pair ``j``) to state ``i``.
Due to the column-major format of Julia, this is also a more efficient representation (in terms of cache locality).

The lower bound is explicitly stored, while the upper bound is computed from the lower bound and the gap. This choice is 
because it simplifies repeated probability assignment using O-maximization [1].

### Fields
- `lower::MR`: The lower bound transition probabilities from a source state or source/action pair to a target state.
- `gap::MR`: The gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
- `sum_lower::VR`: The sum of lower bound transition probabilities from a source state or source/action pair to all target states.

### Examples
```jldoctest
dense_prob = IntervalAmbiguitySets(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

sparse_prob = IntervalAmbiguitySets(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
struct IntervalAmbiguitySets{R, MR <: AbstractMatrix{R}, N, M, I <: LinearIndices} <: AbstractMarginal
    lower::MR
    gap::MR

    state_indices::NTuple{N, Int32}
    action_indices::NTuple{M, Int32}

    source_dims::NTuple{N, Int32}
    action_dims::NTuple{M, Int32}
    linear_index::I

    function IntervalAmbiguitySets(lower::MR, gap::MR, state_indices, action_indices, source_dims, action_dims) where {R, MR <: AbstractMatrix{R}}
        checkprobabilities!(lower, gap)

        linear_index = LinearIndices((source_dims..., action_dims...))
        return IntervalAmbiguitySets(lower, gap, state_indices, action_indices, source_dims, action_dims, linear_index)
    end
end

# Constructor for upper and lower bounds
# Constructor if no state/action indices are given (i.e. only one state and one action variable)

function checkprobabilities!(lower::AbstractMatrix, gap::AbstractMatrix)
    @assert all(lower .>= 0) "The lower bound transition probabilities must be non-negative."
    @assert all(gap .>= 0) "The gap transition probabilities must be non-negative."
    @assert all(lower .+ gap .<= 1) "The sum of lower and gap transition probabilities must be less than or equal to 1."

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    @assert max_lower_bound <= 1 "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."

    sum_upper = vec(sum(lower + gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    @assert max_upper_bound >= 1 "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."
end

function checkprobabilities!(lower::AbstractSparseMatrix, gap::AbstractSparseMatrix)
    @assert all(nonzeros(lower) .>= 0) "The lower bound transition probabilities must be non-negative."
    @assert all(nonzeros(gap) .>= 0) "The gap transition probabilities must be non-negative."
    @assert all(nonzeros(lower) .+ nonzeros(gap) .<= 1) "The sum of lower and gap transition probabilities must be less than or equal to 1."

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    @assert max_lower_bound <= 1 "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."

    sum_upper = vec(sum(lower + gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    @assert max_upper_bound >= 1 "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."
end

function checkindices(
    state_indices::NTuple{N, Int32},
    action_indices::NTuple{M, Int32},
    source_dims::NTuple{N, Int32},
    action_dims::NTuple{M, Int32},
) where {N, M}
    # TODO: More checks
    @assert all(state_indices .> 0) "State indices must be positive."
    @assert all(action_indices .> 0) "Action indices must be positive."

    @assert length(state_indices) == length(source_dims) "Length of state indices must match length of source dimensions."
    @assert length(action_indices) == length(action_dims) "Length of action indices must match length of action dimensions."

    total_source = prod(source_dims)
    total_action = prod(action_dims)

    @assert all(state_indices .<= total_source) "State indices must not exceed total number of source states ($total_source)."
    @assert all(action_indices .<= total_action) "Action indices must not exceed total number of actions ($total_action)."
end

# Keyword constructor from lower and upper
function IntervalAmbiguitySets(; lower::MR, upper::MR) where {MR <: AbstractMatrix}
    lower, gap = compute_gap(lower, upper)
    return IntervalAmbiguitySets(lower, gap)
end

function compute_gap(lower::MR, upper::MR) where {MR <: AbstractMatrix}
    gap = upper - lower
    return lower, gap
end

function compute_gap(
    lower::MR,
    upper::MR,
) where {R, MR <: SparseArrays.AbstractSparseMatrixCSC{R}}
    I, J, _ = findnz(upper)

    gap_nonzeros = Vector{R}(undef, length(I))
    lower_nonzeros = Vector{R}(undef, length(I))

    for (k, (i, j)) in enumerate(zip(I, J))
        gap_nonzeros[k] = upper[i, j] - lower[i, j]
        lower_nonzeros[k] = lower[i, j]
    end

    gap = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        gap_nonzeros,
    )
    lower = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        lower_nonzeros,
    )
    return lower, gap
end

state_variables(p::IntervalAmbiguitySets) = p.source_indices
action_variables(p::IntervalAmbiguitySets) = p.action_indices
source_shape(p::IntervalAmbiguitySets) = p.source_dims
action_shape(p::IntervalAmbiguitySets) = p.action_dims
num_target(p::IntervalAmbiguitySets) = size(p.lower, 1)

struct IntervalAmbiguitySet{R, VR <: AbstractVector{R}}
    lower::VR
    gap::VR
end

function Base.getindex(p::IntervalAmbiguitySets, source::CartesianIndex, action::CartesianIndex)
    source = Tuple(source)[p.state_indices]
    action = Tuple(action)[p.action_indices]
    j = p.linear_index[source..., action...]

    # TODO: Consider ways to avoid specifying/allocating self-loops

    # Select by columns only! 
    l = @view p.lower[:, j]
    g = @view p.gap[:, j]

    return IntervalAmbiguitySet(l, g)
end

lower(p::IntervalAmbiguitySet) = p.lower
lower(p::IntervalAmbiguitySet, destination) = p.lower[destination]

upper(p::IntervalAmbiguitySet) = p.lower + p.gap
upper(p::IntervalAmbiguitySet, destination) = p.lower[destination] + p.gap[destination]

gap(p::IntervalAmbiguitySet) = p.gap
gap(p::IntervalAmbiguitySet, destination) = p.gap[destination]

const ColumnView{Tv} = SubArray{Tv, 1, <:AbstractMatrix{Tv}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
support(p::IntervalAmbiguitySet{R, <:ColumnView{R}}) = eachindex(p.lower)

const SparseColumnView{Tv, Ti} = SubArray{Tv, 1, <:AbstractSparseMatrixCSC{Tv, Ti}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
support(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}) = rowvals(p.lower)