"""
    Marginal{A <: AbstractAmbiguitySets, N, M, I <: LinearIndices}

A struct to represent the dependency graph of an fRMDP, namely by subselecting (in `getindex`) the (decomposed)
state and action. Furthermore, the struct is responsible for converting the Cartesian index to a linear index for
the underlying ambiguity sets.

!!! todo
    Describe source_dims

!!! todo
    Add example

"""
struct Marginal{A <: AbstractAmbiguitySets, N, M, I <: LinearIndices}
    ambiguity_sets::A

    state_indices::NTuple{N, Int32}
    action_indices::NTuple{M, Int32}

    source_dims::NTuple{N, Int32}
    action_vars::NTuple{M, Int32}
    linear_index::I
end

function Marginal(
    ambiguity_sets::A,
    state_indices::NTuple{N, Int32},
    action_indices::NTuple{M, Int32},
    source_dims::NTuple{N, Int32},
    action_vars::NTuple{M, Int32},
) where {A <: AbstractAmbiguitySets, N, M}
    checkindices(ambiguity_sets, state_indices, action_indices, source_dims, action_vars)

    linear_index = LinearIndices((action_vars..., source_dims...))
    return Marginal(ambiguity_sets, state_indices, action_indices, source_dims, action_vars, linear_index)
end

function Marginal(
    ambiguity_sets::A,
    state_indices::NTuple{N, <:Integer},
    action_indices::NTuple{M, <:Integer},
    source_dims::NTuple{N, <:Integer},
    action_vars::NTuple{M, <:Integer},
) where {A <: AbstractAmbiguitySets, N, M}
    state_indices_32 = Int32.(state_indices)
    action_indices_32 = Int32.(action_indices)

    source_dims_32 = Int32.(source_dims)
    action_vars_32 = Int32.(action_vars)

    return Marginal(ambiguity_sets, state_indices_32, action_indices_32, source_dims_32, action_vars_32)
end

function Marginal(ambiguity_sets::A, source_dims, action_vars) where {A <: AbstractAmbiguitySets}
    return Marginal(ambiguity_sets, (1,), (1,), source_dims, action_vars)
end

function checkindices(ambiguity_sets, state_indices, action_indices, source_dims, action_vars)
    if length(state_indices) != length(source_dims)
        throw(ArgumentError("Length of state indices must match length of source dimensions."))
    end

    if length(action_indices) != length(action_vars)
        throw(ArgumentError("Length of action indices must match length of action dimensions."))
    end

    if any(state_indices .<= 0)
        throw(ArgumentError("State indices must be positive."))
    end

    if any(action_indices .<= 0)
        throw(ArgumentError("Action indices must be positive."))
    end
    
    if any(source_dims .<= 0)
        throw(ArgumentError("Source dimensions must be positive."))
    end

    if any(action_vars .<= 0)
        throw(ArgumentError("Action dimensions must be positive."))
    end

    if prod(source_dims) * prod(action_vars) != num_sets(ambiguity_sets)
        throw(ArgumentError("The number of ambiguity sets must match the product of source dimensions and action dimensions."))
    end
end

"""
    ambiguity_sets(p::Marginal)

Return the underlying ambiguity sets of the marginal.
"""
ambiguity_sets(p::Marginal) = p.ambiguity_sets

"""
    state_variables(p::Marginal)

Return the state variable indices of the marginal.
"""
state_variables(p::Marginal) = p.state_indices

"""
    action_variables(p::Marginal)

Return the action variable indices of the marginal.
"""
action_variables(p::Marginal) = p.action_indices

"""
    source_shape(p::Marginal)

Return the shape of the source (state) variables of the marginal. The [`FactoredRobustMarkovDecisionProcess`](@ref) 
checks if this is less than or equal to the corresponding state values.
"""
source_shape(p::Marginal) = p.source_dims

"""
    action_shape(p::Marginal)

Return the shape of the action variables of the marginal. The [`FactoredRobustMarkovDecisionProcess`](@ref)
checks if this is equal to the corresponding action values.
"""
action_shape(p::Marginal) = p.action_vars

"""
    num_target(p::Marginal)

Return the number of target states of the marginal.
"""
num_target(p::Marginal) = num_target(ambiguity_sets(p))

"""
    getindex(p::Marginal, action, source)

Get the ambiguity set corresponding to the given `source` (state) and `action`, where 
the relevant indices of `source` and `action` are selected by `p.action_indices` and `p.state_indices` respectively.
The selected index is then converted to a linear index for the underlying ambiguity sets.
"""
Base.getindex(p::Marginal, action, source) = ambiguity_sets(p)[sub2ind(p, action, source)]

sub2ind(p::Marginal, action::CartesianIndex, source::CartesianIndex) = sub2ind(p, Tuple(action), Tuple(source))
function sub2ind(p::Marginal, action::NTuple{M, T}, source::NTuple{N, T}) where {N, M, T <: Integer}
    action = getindex.((action,), p.action_indices)
    source = getindex.((source,), p.state_indices)
    j = p.linear_index[action..., source...]

    return T(j)
end

function showmarginal(io::IO, prefix, marginal::Marginal)
    println(io, prefix, styled"├─ Conditional variables: {magenta:states = $(state_variables(marginal)), actions = $(action_variables(marginal))}")
    showambiguitysets(io, prefix, ambiguity_sets(marginal))
end

function Base.show(io::IO, mime::MIME"text/plain", marginal::Marginal)
    println(io, styled"{code:Marginal}")
    showmarginal(io, "", marginal)
end