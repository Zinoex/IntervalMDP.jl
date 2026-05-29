"""
    AbstractUpdateSequence{N}

Abstract supertype for sequences of source states to update in a Bellman operator
call. A concrete `AbstractUpdateSequence{N}` is an `AbstractVector{CartesianIndex{N}}`
whose elements are the source-state coordinates to sweep over.

The interface required by `bellman!` is the `AbstractVector` interface — `length`,
linear `getindex`, and iteration — which is enough for both `@threadstid` chunking
(it slices via `length`, `firstindex`, and integer indexing) and for CUDA kernels
(host-side `length` for the launch config, device-side linear `getindex` to recover
each state).

Subtypes:
- [`FullUpdateSequence`](@ref) — a lazy iterator over every source state.
"""
abstract type AbstractUpdateSequence{N} <: AbstractVector{CartesianIndex{N}} end

"""
    FullUpdateSequence(shape::NTuple{N,<:Integer})
    FullUpdateSequence(model)

Lazy update sequence covering every source state of a model. Wraps a
`CartesianIndices` of the source shape so iteration and linear indexing are
allocation-free, and the struct is isbits — meaning it can be passed to a CUDA
kernel without an `Adapt.adapt_structure` definition.
"""
struct FullUpdateSequence{N, R <: NTuple{N, AbstractUnitRange{<:Integer}}} <:
       AbstractUpdateSequence{N}
    indices::CartesianIndices{N, R}
end

FullUpdateSequence(shape::Tuple{Vararg{Integer}}) =
    FullUpdateSequence(CartesianIndices(map(Base.OneTo, shape)))
FullUpdateSequence(model) = FullUpdateSequence(source_shape(model))

Base.size(s::FullUpdateSequence) = (length(s.indices),)
Base.IndexStyle(::Type{<:FullUpdateSequence}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(s::FullUpdateSequence, i::Int) = s.indices[i]

"""
    default_update_sequence(model)

Default `AbstractUpdateSequence` used by `bellman!` when the caller does not pass one.
The fallback is `FullUpdateSequence(model)` (every source state). For `ProductProcess`
the default is the full source-state sweep of the *underlying* Markov process — the
DFA-state dimension is handled by the product-process bellman helper itself, so the
sweep argument controls only the underlying mp.
"""
default_update_sequence(model) = FullUpdateSequence(model)
default_update_sequence(model::ProductProcess) =
    FullUpdateSequence(markov_process(model))
