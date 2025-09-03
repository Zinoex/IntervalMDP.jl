struct SARectangularMarginal{A <: AbstractAmbiguitySets, N, M, I <: LinearIndices} <: AbstractMarginal
    ambiguity_sets::A

    state_indices::NTuple{N, Int32}
    action_indices::NTuple{M, Int32}

    source_dims::NTuple{N, Int32}
    action_vars::NTuple{M, Int32}
    linear_index::I

    function SARectangularMarginal(
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
const Marginal = SARectangularMarginal

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

    return SARectangularMarginal(ambiguity_sets, state_indices_32, action_indices_32, source_dims_32, action_vars_32)
end

# Constructor if no state/action indices are given (i.e. only one state and one action variable)

function checkindices(ambiguity_sets, state_indices, action_indices, source_dims, action_vars)
    # TODO: More checks incl. consistency with ambiguity sets
    @assert all(state_indices .> 0) "State indices must be positive."
    @assert all(action_indices .> 0) "Action indices must be positive."

    @assert length(state_indices) == length(source_dims) "Length of state indices must match length of source dimensions."
    @assert length(action_indices) == length(action_vars) "Length of action indices must match length of action dimensions."
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
    action = getindex.(Tuple(action), p.action_indices)
    source = getindex.(Tuple(source), p.state_indices)
    j = p.linear_index[action..., source...]

    return j
end