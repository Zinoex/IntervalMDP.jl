"""
    IntervalAmbiguitySets{R, MR <: AbstractMatrix{R}}

A matrix pair to represent the lower and upper bound of `num_sets(ambiguity_sets)` interval ambiguity sets (on the columns)
to `num_target(ambiguity_sets)` destinations (on the rows). [Marginal](@ref) adds interpretation to the column indices.
The matrices can be `Matrix{R}` or `SparseMatrixCSC{R}`, or their CUDA equivalents. 
Due to the space complexity, if modelling [IntervalMarkovChains](@ref IntervalMarkovChain) or [IntervalMarkovDecisionProcesses](@ref IntervalMarkovDecisionProcess),
it is recommended to use sparse matrices.

The columns represent the different ambiguity sets and the rows represent the targets. Due to the column-major format of Julia, 
this is a more efficient representation in terms of cache locality.

The lower bound is explicitly stored, while the upper bound is computed from the lower bound and the gap. This choice is 
because it simplifies repeated probability assignment using O-maximization [givan2000bounded, lahijanian2015formal](@cite).

### Fields
- `lower::MR`: The lower bound probabilities for `num_sets(ambiguity_sets)` ambiguity sets to `num_target(ambiguity_sets)` target states.
- `gap::MR`: The gap between upper and lower bound transition probabilities for `num_sets(ambiguity_sets)` ambiguity sets to `num_target(ambiguity_sets)` target states.

### Examples
```jldoctest
using IntervalMDP

dense_prob = IntervalAmbiguitySets(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

# output

IntervalAmbiguitySets
├─ Storage type: Matrix{Float64}
├─ Number of target states: 3
└─ Number of ambiguity sets: 2
```

```jldoctest
using IntervalMDP, SparseArrays
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

# output

IntervalAmbiguitySets
├─ Storage type: SparseArrays.FixedSparseCSC{Float64, Int64}
├─ Number of target states: 15
├─ Number of ambiguity sets: 2
├─ Maximum support size: 3
└─ Number of non-zeros: 6
```

"""
struct IntervalAmbiguitySets{R, MR <: AbstractMatrix{R}} <: PolytopicAmbiguitySets
    lower::MR
    gap::MR

    function IntervalAmbiguitySets(
        lower::MR,
        gap::MR,
        check::Val{true},
    ) where {R, MR <: AbstractMatrix{R}}
        checkprobabilities(lower, gap)

        return new{R, MR}(lower, gap)
    end

    function IntervalAmbiguitySets(
        lower::MR,
        gap::MR,
        check::Val{false},
    ) where {R, MR <: AbstractMatrix{R}}
        return new{R, MR}(lower, gap)
    end
end

IntervalAmbiguitySets(lower::MR, gap::MR) where {R, MR <: AbstractMatrix{R}} =
    IntervalAmbiguitySets(lower, gap, Val(true))

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
    if size(lower) != size(upper)
        throw(DimensionMismatch("The lower and upper matrices must have the same size."))
    end

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
        throw(DimensionMismatch("The lower and gap matrices must have the same size."))
    end

    if any(lower .< 0)
        throw(
            ArgumentError("The lower bound transition probabilities must be non-negative."),
        )
    end

    if any(lower .> 1)
        throw(
            ArgumentError(
                "The lower bound transition probabilities must be less than or equal to 1.",
            ),
        )
    end

    if any(gap .< 0)
        throw(ArgumentError("The gap transition probabilities must be non-negative."))
    end

    if any(lower .+ gap .> 1)
        throw(
            ArgumentError(
                "The sum of lower and gap transition probabilities must be less than or equal to 1.",
            ),
        )
    end

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    if max_lower_bound > 1
        throw(
            ArgumentError(
                "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1.",
            ),
        )
    end

    sum_upper = sum_lower .+ vec(sum(gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    if max_upper_bound < 1
        throw(
            ArgumentError(
                "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1.",
            ),
        )
    end
end

function checkprobabilities(lower::MR, gap::MR) where {R, MR <: AbstractSparseMatrix{R}}
    if size(lower) != size(gap)
        throw(DimensionMismatch("The lower and gap matrices must have the same size."))
    end

    if SparseArrays.getcolptr(lower) != SparseArrays.getcolptr(gap)
        throw(
            DimensionMismatch(
                "The lower and gap matrices must have the same column structure.",
            ),
        )
    end

    if SparseArrays.rowvals(lower) != SparseArrays.rowvals(gap)
        throw(
            DimensionMismatch(
                "The lower and gap matrices must have the same row structure.",
            ),
        )
    end

    if any(nonzeros(lower) .< 0)
        throw(
            ArgumentError("The lower bound transition probabilities must be non-negative."),
        )
    end

    if any(nonzeros(lower) .> 1)
        throw(
            ArgumentError(
                "The lower bound transition probabilities must be less than or equal to 1.",
            ),
        )
    end

    if any(nonzeros(gap) .< 0)
        throw(ArgumentError("The gap transition probabilities must be non-negative."))
    end

    if any(nonzeros(gap) .> 1)
        throw(
            ArgumentError(
                "The gap transition probabilities must be less than or equal to 1.",
            ),
        )
    end

    if any(nonzeros(lower) .+ nonzeros(gap) .> 1)
        throw(
            ArgumentError(
                "The sum of lower and gap transition probabilities must be less than or equal to 1.",
            ),
        )
    end

    sum_lower = vec(sum(lower; dims = 1))
    max_lower_bound = maximum(sum_lower)
    if max_lower_bound > 1
        throw(
            ArgumentError(
                "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1.",
            ),
        )
    end

    sum_upper = sum_lower .+ vec(sum(gap; dims = 1))
    max_upper_bound = minimum(sum_upper)
    if max_upper_bound < 1
        throw(
            ArgumentError(
                "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1.",
            ),
        )
    end
end

num_target(p::IntervalAmbiguitySets) = size(p.lower, 1)
num_sets(p::IntervalAmbiguitySets) = size(p.lower, 2)

source_shape(p::IntervalAmbiguitySets) = (num_sets(p),)
action_shape(::IntervalAmbiguitySets) = (1,)
marginals(p::IntervalAmbiguitySets) = (p,)
available_actions(::IntervalAmbiguitySets) = AllAvailableActions((1,))

maxsupportsize(p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractMatrix{R}} =
    size(p.gap, 1)
maxsupportsize(
    p::IntervalAmbiguitySets{R, MR},
) where {R, MR <: SparseArrays.AbstractSparseMatrixCSC{R}} =
    maxdiff(SparseArrays.getcolptr(p.gap))

Base.@propagate_inbounds function Base.getindex(p::IntervalAmbiguitySets, j::Integer)
    # Select by columns only! 
    l = @view p.lower[:, j]
    g = @view p.gap[:, j]

    return IntervalAmbiguitySet(l, g)
end

Base.@propagate_inbounds sub2ind(
    ::IntervalAmbiguitySets,
    jₐ::NTuple{M, T},
    jₛ::NTuple{N, T},
) where {N, M, T <: Integer} = T(jₛ[1])
Base.@propagate_inbounds sub2ind(p::IntervalAmbiguitySets, jₐ::CartesianIndex, jₛ::CartesianIndex) =
    sub2ind(p, Tuple(jₐ), Tuple(jₛ))
Base.@propagate_inbounds Base.getindex(p::IntervalAmbiguitySets, jₐ, jₛ) = p[sub2ind(p, jₐ, jₛ)]

Base.iterate(p::IntervalAmbiguitySets) = (p[1], 2)
function Base.iterate(p::IntervalAmbiguitySets, state)
    if state > num_sets(p)
        return nothing
    else
        return (p[state], state + 1)
    end
end
Base.length(p::IntervalAmbiguitySets) = num_sets(p)

function showambiguitysets(
    io::IO,
    prefix,
    ::IntervalAmbiguitySets{R, MR},
) where {R, MR <: AbstractMatrix}
    println(io, prefix, styled"└─ Ambiguity set type: Interval (dense, {code:$MR})")
end

function showambiguitysets(
    io::IO,
    prefix,
    p::IntervalAmbiguitySets{R, MR},
) where {R, MR <: AbstractSparseMatrix}
    println(io, prefix, styled"├─ Ambiguity set type: Interval (sparse, {code:$MR})")
    num_transitions = nnz(p.gap)
    max_support = maxsupportsize(p)
    println(
        io,
        prefix,
        styled"└─ Transitions: {magenta: $num_transitions (max support: $max_support)}",
    )
end

function Base.show(
    io::IO,
    mime::MIME"text/plain",
    p::IntervalAmbiguitySets{R, MR},
) where {R, MR <: AbstractMatrix}
    println(io, styled"{code:IntervalAmbiguitySets}")
    println(io, styled"├─ Storage type: {code:$MR}")
    println(io, "├─ Number of target states: ", num_target(p))
    println(io, "└─ Number of ambiguity sets: ", num_sets(p))
end

function Base.show(
    io::IO,
    mime::MIME"text/plain",
    p::IntervalAmbiguitySets{R, MR},
) where {R, MR <: AbstractSparseMatrix}
    println(io, styled"{code:IntervalAmbiguitySets}")
    println(io, styled"├─ Storage type: {code:$MR}")
    println(io, "├─ Number of target states: ", num_target(p))
    println(io, "├─ Number of ambiguity sets: ", num_sets(p))
    println(io, "├─ Maximum support size: ", maxsupportsize(p))
    println(io, "└─ Number of non-zeros: ", nnz(p.gap))
end

struct IntervalAmbiguitySet{R, VR <: AbstractVector{R}} <: PolytopicAmbiguitySet
    lower::VR
    gap::VR
end

num_target(p::IntervalAmbiguitySet) = length(p.lower)

"""
    lower(p::IntervalAmbiguitySet)

Return the lower bound transition probabilities of the ambiguity set to all target states.
"""
lower(p::IntervalAmbiguitySet) = p.lower
Base.@propagate_inbounds lower(p::IntervalAmbiguitySet, destination) = p.lower[destination]

"""
    upper(p::IntervalAmbiguitySet)

Return the upper bound transition probabilities of the ambiguity set to all target states.
"""
upper(p::IntervalAmbiguitySet) = p.lower + p.gap
Base.@propagate_inbounds upper(p::IntervalAmbiguitySet, destination) = p.lower[destination] + p.gap[destination]

"""
    gap(p::IntervalAmbiguitySet)

Return the gap between upper and lower bound transition probabilities of the ambiguity set to all target states.
"""
gap(p::IntervalAmbiguitySet) = p.gap
Base.@propagate_inbounds gap(p::IntervalAmbiguitySet, destination) = p.gap[destination]

const ColumnView{Tv} =
    SubArray{Tv, 1, <:AbstractMatrix{Tv}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
support(p::IntervalAmbiguitySet{R, <:ColumnView{R}}) where {R} = eachindex(p.gap)
support(::IntervalAmbiguitySet{R, <:ColumnView{R}}, s) where {R} = s
supportsize(p::IntervalAmbiguitySet{R, <:ColumnView{R}}) where {R} = Int32(length(p.gap))

const SparseColumnView{Tv, Ti} = SubArray{
    Tv,
    1,
    <:SparseArrays.AbstractSparseMatrixCSC{Tv, Ti},
    Tuple{Base.Slice{Base.OneTo{Int}}, Int},
    false
}
Base.@propagate_inbounds support(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}) where {R} = rowvals(p.gap)
Base.@propagate_inbounds support(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}, s) where {R} = support(p)[s]
Base.@propagate_inbounds supportsize(p::IntervalAmbiguitySet{R, <:SparseColumnView{R}}) where {R} = Int32(nnz(p.gap))

# Vertex iterator for IntervalAmbiguitySet
struct IntervalAmbiguitySetVertexIterator{R, VR <: AbstractVector{R}} <: VertexIterator
    set::IntervalAmbiguitySet{R, VR}
    result::Vector{R}  # Preallocated result vector
end

function IntervalAmbiguitySetVertexIterator(set::IntervalAmbiguitySet)
    v = Vector{valuetype(set)}(undef, num_target(set))
    return IntervalAmbiguitySetVertexIterator(set, v)
end

Base.IteratorSize(::Type{<:IntervalAmbiguitySetVertexIterator}) = Base.SizeUnknown()

function Base.iterate(
    it::IntervalAmbiguitySetVertexIterator{R, VR},
) where {R, VR <: AbstractVector{R}}
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

function Base.iterate(
    it::IntervalAmbiguitySetVertexIterator{R, VR},
    state,
) where {R, VR <: AbstractVector{R}}
    (permutation, last_break_idx) = state

    # Skip permutations that would lead to the same vertex
    # based on the prefix 1:last_break_idx
    break_j = nothing
    for j in last_break_idx:-1:1
        # Find smallest permutation[k] in permutation[j+1:end] where permutation[j] < permutation[k]
        next_in_suffix = nothing
        for k in (j + 1):length(permutation)
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
        permutation[j], permutation[next_in_suffix] =
            permutation[next_in_suffix], permutation[j]
        break_j = j
        break
    end

    if isnothing(break_j)
        return nothing
    end

    sort!(@view(permutation[(break_j + 1):end]))

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
vertex_generator(p::IntervalAmbiguitySet, result::Vector) =
    IntervalAmbiguitySetVertexIterator(p, result)
vertices(p::IntervalAmbiguitySet) = map(copy, vertex_generator(p))
