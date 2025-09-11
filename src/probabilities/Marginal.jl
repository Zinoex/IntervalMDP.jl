struct Marginal{A <: AbstractAmbiguitySets, N, M, I <: LinearIndices}
    ambiguity_sets::A

    state_indices::NTuple{N, Int32}
    action_indices::NTuple{M, Int32}

    source_dims::NTuple{N, Int32}
    action_vars::NTuple{M, Int32}
    linear_index::I

    function Marginal(
        ambiguity_sets::A,
        state_indices::NTuple{N, Int32},
        action_indices::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
    ) where {A <: AbstractAmbiguitySets, N, M}
        checkindices(ambiguity_sets, state_indices, action_indices, source_dims, action_vars)

        linear_index = LinearIndices((action_vars..., source_dims...))
        return new{A, N, M, typeof(linear_index)}(ambiguity_sets, state_indices, action_indices, source_dims, action_vars, linear_index)
    end
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

ambiguity_sets(p::Marginal) = p.ambiguity_sets
state_variables(p::Marginal) = p.state_indices
action_variables(p::Marginal) = p.action_indices
source_shape(p::Marginal) = p.source_dims
action_shape(p::Marginal) = p.action_vars
num_target(p::Marginal) = num_target(ambiguity_sets(p))

function Base.getindex(p::Marginal, source, action)
    return ambiguity_sets(p)[sub2ind(p, source, action)]
end

sub2ind(p::Marginal, action::CartesianIndex, source::CartesianIndex) = sub2ind(p, Tuple(action), Tuple(source))
function sub2ind(p::Marginal, action::NTuple{M, <:Integer}, source::NTuple{N, <:Integer}) where {N, M}
    action = getindex.((action,), p.action_indices)
    source = getindex.((source,), p.state_indices)
    j = p.linear_index[action..., source...]

    return j
end