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
struct Marginal{A <: AbstractAmbiguitySets, N, M}
    ambiguity_sets::A

    state_indices::NTuple{N, Int32}
    action_indices::NTuple{M, Int32}

    source_dims::NTuple{N, Int32}
    action_vars::NTuple{M, Int32}

    function Marginal(
        ambiguity_sets::A,
        state_indices::NTuple{N, Int32},
        action_indices::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
        check::Val{false}
    ) where {A <: AbstractAmbiguitySets, N, M}
        new{A, N, M}(
            ambiguity_sets,
            state_indices,
            action_indices,
            source_dims,
            action_vars,
        )
    end

    function Marginal(
        ambiguity_sets::A,
        state_indices::NTuple{N, Int32},
        action_indices::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
        check::Val{true},
    ) where {A <: AbstractAmbiguitySets, N, M}
        checkindices(ambiguity_sets, state_indices, action_indices, source_dims, action_vars)

        return new{A, N, M}(
            ambiguity_sets,
            state_indices,
            action_indices,
            source_dims,
            action_vars,
        )
    end
end

function Marginal(
    ambiguity_sets::A,
    state_indices::NTuple{N, Int32},
    action_indices::NTuple{M, Int32},
    source_dims::NTuple{N, Int32},
    action_vars::NTuple{M, Int32},
) where {A <: AbstractAmbiguitySets, N, M}
    return Marginal(
        ambiguity_sets,
        state_indices,
        action_indices,
        source_dims,
        action_vars,
        Val(true),
    )
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

    return Marginal(
        ambiguity_sets,
        state_indices_32,
        action_indices_32,
        source_dims_32,
        action_vars_32,
    )
end

function Marginal(
    ambiguity_sets::A,
    source_dims,
    action_vars,
) where {A <: AbstractAmbiguitySets}
    return Marginal(ambiguity_sets, (1,), (1,), source_dims, action_vars)
end

function checkindices(
    ambiguity_sets,
    state_indices,
    action_indices,
    source_dims,
    action_vars,
)
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
        throw(
            ArgumentError(
                "The number of ambiguity sets must match the product of source dimensions and action dimensions.",
            ),
        )
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
Base.@propagate_inbounds Base.getindex(p::Marginal, action, source) = ambiguity_sets(p)[sub2ind(p, action, source)]

Base.@propagate_inbounds sub2ind(p::Marginal, action::CartesianIndex, source::CartesianIndex) =
    sub2ind(p, Tuple(action), Tuple(source))
Base.@propagate_inbounds function sub2ind(
    p::Marginal{A, N1, M1},
    action::NTuple{M2, T},
    source::NTuple{N2, T},
) where {A, N1, M1, N2, M2, T <: Integer}
    ind = zero(T)

    for i in StepRange(N1, -1, 1)
        ind *= p.source_dims[i]
        ind += source[p.state_indices[i]] - one(T)
    end

    for i in StepRange(M1, -1, 1)
        ind *= p.action_vars[i]
        ind += action[p.action_indices[i]] - one(T)
    end

    return ind + one(T)
end

function showmarginal(io::IO, prefix, marginal::Marginal)
    println(
        io,
        prefix,
        styled"├─ Conditional variables: {magenta:states = $(state_variables(marginal)), actions = $(action_variables(marginal))}",
    )
    showambiguitysets(io, prefix, ambiguity_sets(marginal))
end

function Base.show(io::IO, mime::MIME"text/plain", marginal::Marginal)
    println(io, styled"{code:Marginal}")
    showmarginal(io, "", marginal)
end
