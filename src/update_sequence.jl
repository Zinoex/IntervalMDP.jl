"""
    AbstractUpdateSequence

Abstract supertype for objects that describe *which* source states a Bellman
operator call should update.

There are two flavours:

- [`FullUpdateSequence`](@ref) and other "flat" sequences: a vector-like
  collection of `CartesianIndex` values over the source space of an
  [`IntervalMarkovProcess`](@ref). Flat sequences implement the standard
  `AbstractVector` interface (`length`, linear `getindex`, iteration,
  `firstindex`/`lastindex`) so they slot directly into `for jₛ in states`
  loops, the `@threadstid` chunking macro, and CUDA kernels.
- [`ProductUpdateSequence`](@ref): a structured pair `(dfa_states, mp_states)`
  for [`ProductProcess`](@ref) models, where `mp_states` may either be a single
  flat sequence shared by every DFA state or an indexed collection that varies
  per DFA state.

Subtypes are not required to subtype `AbstractVector` — the product variant
does not, because flattening "(dfa, mp)" to a single vector would lose the
structural distinction the product-process Bellman helper relies on.
"""
abstract type AbstractUpdateSequence end

"""
    FullUpdateSequence(shape::Tuple{Vararg{Integer}})
    FullUpdateSequence(model)

Lazy flat update sequence covering every source state. Wraps a `CartesianIndices`
over the source shape so iteration and linear indexing are allocation-free. The
struct is isbits, so it can be passed to a CUDA kernel without an
`Adapt.adapt_structure` definition.

Implements the `AbstractVector{CartesianIndex{N}}` interface — `length`, linear
`getindex`, iteration, `firstindex`/`lastindex`, `IndexLinear` style — directly,
without subtyping `AbstractVector`. This keeps the [`AbstractUpdateSequence`](@ref)
hierarchy free of the `AbstractArray` contract that does not fit the structured
[`ProductUpdateSequence`](@ref).
"""
struct FullUpdateSequence{N, R <: NTuple{N, AbstractUnitRange{<:Integer}}} <:
       AbstractUpdateSequence
    indices::CartesianIndices{N, R}
end

FullUpdateSequence(shape::Tuple{Vararg{Integer}}) =
    FullUpdateSequence(CartesianIndices(map(Base.OneTo, shape)))
FullUpdateSequence(model) = FullUpdateSequence(source_shape(model))

Base.length(s::FullUpdateSequence) = length(s.indices)
Base.size(s::FullUpdateSequence) = (length(s.indices),)
Base.firstindex(::FullUpdateSequence) = 1
Base.lastindex(s::FullUpdateSequence) = length(s)
Base.IndexStyle(::Type{<:FullUpdateSequence}) = IndexLinear()
Base.eltype(::Type{<:FullUpdateSequence{N}}) where {N} = CartesianIndex{N}
Base.@propagate_inbounds Base.getindex(s::FullUpdateSequence, i::Integer) =
    s.indices[i]
Base.@propagate_inbounds function Base.iterate(s::FullUpdateSequence, state::Int = 1)
    state > length(s) && return nothing
    return (@inbounds s[state], state + 1)
end

"""
    ProductUpdateSequence(dfa_states, mp_states)

Update sequence for a [`ProductProcess`](@ref). The product-process Bellman
helper sweeps over the DFA dimension explicitly and dispatches to the
underlying Markov-process Bellman for each DFA state; this type carries the
two pieces separately:

- `dfa_states::AbstractUpdateSequence` — the DFA states to update. Typically a
  [`FullUpdateSequence`](@ref) of `source_shape(automaton(model))`, but any
  flat sequence of `CartesianIndex{1}` works (e.g. for skipping terminal
  states).
- `mp_states::M` — the source-state sweep for the underlying Markov process.
  May be:
    * a single `AbstractUpdateSequence` (same sweep for every DFA state — the
      default), or
    * an `AbstractVector{<:AbstractUpdateSequence}` indexed by DFA-state
      integer (one sweep per DFA state).

The accessor `mp_sequence(p, dfa_state)` returns the mp sub-sequence used for
the given DFA state, dispatching on the kind of `mp_states`.
"""
struct ProductUpdateSequence{D <: AbstractUpdateSequence, M} <: AbstractUpdateSequence
    dfa_states::D
    mp_states::M
end

"""
    mp_sequence(p::ProductUpdateSequence, dfa_state)

Return the underlying Markov-process update sequence that the product-process
Bellman helper should use for `dfa_state`. Falls back to the single shared
`mp_states` when one is stored; indexes into the per-DFA-state collection
otherwise.
"""
mp_sequence(p::ProductUpdateSequence, dfa_state) = _mp_sequence(p.mp_states, dfa_state)
_mp_sequence(seq::AbstractUpdateSequence, _) = seq
Base.@propagate_inbounds _mp_sequence(
    seqs::AbstractVector{<:AbstractUpdateSequence},
    dfa_state,
) = seqs[dfa_state]

"""
    default_update_sequence(model)

Default `AbstractUpdateSequence` used by `bellman!` when the caller does not
pass one. The fallback is `FullUpdateSequence(model)` (every source state).
For a [`ProductProcess`](@ref) the default is a [`ProductUpdateSequence`](@ref)
combining a full DFA sweep with a single full mp sweep shared across every DFA
state.
"""
default_update_sequence(model) = FullUpdateSequence(model)
default_update_sequence(model::ProductProcess) = ProductUpdateSequence(
    FullUpdateSequence(automaton(model)),
    FullUpdateSequence(markov_process(model)),
)
