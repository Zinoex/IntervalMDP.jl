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
struct IntervalAmbiguitySets{R, MR <: AbstractMatrix{R}} <: PolytopicAmbiguitySets
    lower::MR
    gap::MR

    function IntervalAmbiguitySets(lower::MR, gap::MR, check::Val{true}) where {R, MR <: AbstractMatrix{R}}
        checkprobabilities(lower, gap)

        return new{R, MR}(lower, gap)
    end

    function IntervalAmbiguitySets(lower::MR, gap::MR, check::Val{false}) where {R, MR <: AbstractMatrix{R}}
        return new{R, MR}(lower, gap)
    end
end

IntervalAmbiguitySets(lower::MR, gap::MR) where {R, MR <: AbstractMatrix{R}} = IntervalAmbiguitySets(lower, gap, Val(true))

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

function checkprobabilities(lower::AbstractMatrix, gap::AbstractMatrix)
    if size(lower) != size(gap)
        throw(ArgumentError("The lower and gap matrices must have the same size."))
    end

    if any(lower .< 0)
        throw(ArgumentError("The lower bound transition probabilities must be non-negative."))
    end

    if any(lower .> 1)
        throw(ArgumentError("The lower bound transition probabilities must be less than or equal to 1."))
    end

    if any(gap .< 0)
        throw(ArgumentError("The gap transition probabilities must be non-negative."))
    end

    if any(gap .> 1)
        throw(ArgumentError("The gap transition probabilities must be less than or equal to 1."))
    end

    if any(lower .+ gap .> 1)
        throw(ArgumentError("The sum of lower and gap transition probabilities must be less than or equal to 1."))
    end

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    if max_lower_bound > 1
        throw(ArgumentError("The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."))
    end

    sum_upper = sum_lower .+ vec(sum(gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    if max_upper_bound < 1
        throw(ArgumentError("The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."))
    end
end

function checkprobabilities!(lower::AbstractSparseMatrix, gap::AbstractSparseMatrix)
    if size(lower) != size(gap)
        throw(ArgumentError("The lower and gap matrices must have the same size."))
    end

    if any(nonzeros(lower) .< 0)
        throw(ArgumentError("The lower bound transition probabilities must be non-negative."))
    end

    if any(nonzeros(lower) .> 1)
        throw(ArgumentError("The lower bound transition probabilities must be less than or equal to 1."))
    end

    if any(nonzeros(gap) .< 0)
        throw(ArgumentError("The gap transition probabilities must be non-negative."))
    end

    if any(nonzeros(gap) .> 1)
        throw(ArgumentError("The gap transition probabilities must be less than or equal to 1."))
    end

    if any(nonzeros(lower) .+ nonzeros(gap) .> 1)
        throw(ArgumentError("The sum of lower and gap transition probabilities must be less than or equal to 1."))
    end

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    if max_lower_bound > 1
        throw(ArgumentError("The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."))
    end

    sum_upper = sum_lower .+ vec(sum(gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    if max_upper_bound < 1
        throw(ArgumentError("The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."))
    end
end

num_target(p::IntervalAmbiguitySets) = size(p.lower, 1)
num_sets(p::IntervalAmbiguitySets) = size(p.lower, 2)
source_shape(p::IntervalAmbiguitySets) = (num_sets(p),)
action_shape(::IntervalAmbiguitySets) = (1,)
marginals(p::IntervalAmbiguitySets) = (p,)

function Base.getindex(p::IntervalAmbiguitySets, j::Integer)
    # Select by columns only! 
    l = @view p.lower[:, j]
    g = @view p.gap[:, j]

    return IntervalAmbiguitySet(l, g)
end

sub2ind(::IntervalAmbiguitySets, jₐ::NTuple{M, T}, jₛ::NTuple{N, T}) where {N, M, T <: Integer} = T(jₛ[1])
sub2ind(p::IntervalAmbiguitySets, jₐ::CartesianIndex, jₛ::CartesianIndex) = sub2ind(p, Tuple(jₐ), Tuple(jₛ))
Base.getindex(p::IntervalAmbiguitySets, jₐ, jₛ) = p[sub2ind(p, jₐ, jₛ)]

Base.iterate(p::IntervalAmbiguitySets) = (p[1], 2)
function Base.iterate(p::IntervalAmbiguitySets, state)
    if state > num_sets(p)
        return nothing
    else
        return (p[state], state + 1)
    end
end
Base.length(p::IntervalAmbiguitySets) = num_sets(p)

function showambiguitysets(io::IO, prefix, ::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractMatrix}
    println(io, prefix, styled"└─ Ambiguity set type: Interval (dense, {code:$MR})")
end

function showambiguitysets(io::IO, prefix, p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractSparseMatrix}
    println(io, prefix, styled"├─ Ambiguity set type: Interval (sparse, {code:$MR})")
    num_transitions = nnz(p.gap)
    max_support = maximum(supportsize, p)
    println(io, prefix, styled"└─ Transitions: {magenta: $num_transitions (max support: $max_support)}")
end

function Base.show(io::IO, mime::MIME"text/plain", p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractMatrix}
    println(io, styled"{code:IntervalAmbiguitySets}")
    println(io, styled"├─ Storage type: {code:$MR}")
    println(io, "├─ Number of target states: ", num_target(p))
    println(io, "└─ Number of ambiguity sets: ", num_sets(p))
end

function Base.show(io::IO, mime::MIME"text/plain", p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractSparseMatrix}
    println(io, styled"{code:IntervalAmbiguitySets}")
    println(io, styled"├─ Storage type: {code:$MR}")
    println(io, "├─ Number of target states: ", num_target(p))
    println(io, "├─ Number of ambiguity sets: ", num_sets(p))
    println(io, "├─ Maximum support size: ", maximum(supportsize, p))
    println(io, "└─ Number of non-zeros: ", nnz(p.gap))
end

struct IntervalAmbiguitySet{R, VR <: AbstractVector{R}} <: PolytopicAmbiguitySet
    lower::VR
    gap::VR
end

num_target(p::IntervalAmbiguitySet) = length(p.lower)

lower(p::IntervalAmbiguitySet) = p.lower
lower(p::IntervalAmbiguitySet, destination) = p.lower[destination]

upper(p::IntervalAmbiguitySet) = p.lower + p.gap
upper(p::IntervalAmbiguitySet, destination) = p.lower[destination] + p.gap[destination]

gap(p::IntervalAmbiguitySet) = p.gap
gap(p::IntervalAmbiguitySet, destination) = p.gap[destination]

const ColumnView{Tv} = SubArray{Tv, 1, <:AbstractMatrix{Tv}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
support(p::IntervalAmbiguitySet{R, <:ColumnView{R}}) where {R} = eachindex(p.gap)
supportsize(p::IntervalAmbiguitySet{R, <:ColumnView{R}}) where {R} = length(p.gap)

const SparseColumnView{Tv, Ti} = SubArray{Tv, 1, <:SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
support(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}) where {R} = rowvals(p.gap)
supportsize(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}) where {R} = nnz(p.gap)

# Vertex iterator for IntervalAmbiguitySet
struct IntervalAmbiguitySetVertexIterator{R, VR <: AbstractVector{R}} <: VertexIterator
    set::IntervalAmbiguitySet{R, VR}
    result::Vector{R}  # Preallocated result vector
end

function IntervalAmbiguitySetVertexIterator(set::IntervalAmbiguitySet)
    v = Vector{valuetype(set)}(undef, num_target(set))
    return IntervalAmbiguitySetVertexIterator(set, v)
end

Base.IteratorEltype(::Type{<:IntervalAmbiguitySetVertexIterator}) = Base.HasEltype()
Base.eltype(::IntervalAmbiguitySetVertexIterator{R}) where {R} = Vector{R}
Base.IteratorSize(::Type{<:IntervalAmbiguitySetVertexIterator}) = Base.SizeUnknown()

function Base.iterate(it::IntervalAmbiguitySetVertexIterator{R, VR}) where {R, VR <: AbstractVector{R}}
    permutation = collect(1:length(support(it.set)))

    v = it.result
    copyto!(v, lower(it.set))
    budget = one(R) - sum(v)

    break_idx = 0
    for (j, i) in enumerate(permutation)
        i = support(it.set)[i]
        if budget <= gap(it.set, i)
            v[i] += budget
            break_idx = j
            break
        else
            v[i] += gap(it.set, i)
            budget -= gap(it.set, i)
        end
    end

    return v, (permutation, break_idx)
end

function Base.iterate(it::IntervalAmbiguitySetVertexIterator{R, VR}, state) where {R, VR <: AbstractVector{R}}
    (permutation, last_break_idx) = state

    # Skip permutations that would lead to the same vertex
    # based on the prefix 1:last_break_idx
    break_j = nothing
    for j in last_break_idx:-1:1
        # Find smallest permutation[k] in permutation[j+1:end] where permutation[j] < permutation[k]
        next_in_suffix = nothing
        for k in j+1:length(permutation)
            if permutation[k] > permutation[j]
                if isnothing(next_in_suffix) || permutation[k] < permutation[next_in_suffix]
                    next_in_suffix = k
                end
            end
        end

        if isnothing(next_in_suffix) # No such k exists, continue to next j
            continue
        end

        # Swap
        permutation[j], permutation[next_in_suffix] = permutation[next_in_suffix], permutation[j]
        break_j = j
        break
    end

    if isnothing(break_j)
        return nothing
    end

    sort!(@view(permutation[break_j+1:end]))

    # Now compute the vertex for this new permutation
    v = it.result
    copyto!(v, lower(it.set))
    budget = one(R) - sum(v)

    if iszero(budget)
        return nothing
    end

    break_idx = 0
    for (j, i) in enumerate(permutation)
        i = support(it.set)[i]
        if budget <= gap(it.set, i)
            v[i] += budget
            break_idx = j
            break
        else
            v[i] += gap(it.set, i)
            budget -= gap(it.set, i)
        end
    end

    return v, (permutation, break_idx)
end

vertex_generator(p::IntervalAmbiguitySet) = IntervalAmbiguitySetVertexIterator(p)
vertex_generator(p::IntervalAmbiguitySet, result::Vector) = IntervalAmbiguitySetVertexIterator(p, result)
vertices(p::IntervalAmbiguitySet) = map(copy, vertex_generator(p))