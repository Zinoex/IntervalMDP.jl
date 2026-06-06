abstract type TerminationCriteria end

struct FixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::FixedIterationsCriteria)(V, k, u) = k >= f.n

struct CovergenceCriteria{T <: Real} <: TerminationCriteria
    tol::T
end
(f::CovergenceCriteria)(V, k, u) = maximum(abs, u) < f.tol

"""
    InitialStateCriteria(inner, initial)

Restrict a convergence-style termination criterion `inner` to a subset of state
indices `initial` (the system's initial states). The residual `u` is indexed by
`initial` before being handed to `inner`, so convergence is decided only on those
states. Composes with both residual-based criteria (e.g. [`CovergenceCriteria`],
used by [`RobustValueIteration`](@ref)) and gap-based criteria (used by
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref)).
"""
struct InitialStateCriteria{C <: TerminationCriteria, I} <: TerminationCriteria
    inner::C
    initial::I
end
(f::InitialStateCriteria)(V, k, u) = f.inner(V, k, @view u[f.initial])

# Base criterion for residual/iteration-based value iteration: fixed iterations
# for finite-time properties, convergence on the residual for infinite-time.
base_termination_criteria(prop::Property) =
    isfinitetime(prop) ? FixedIterationsCriteria(time_horizon(prop)) :
    CovergenceCriteria(convergence_eps(prop))

# Wrap a base criterion so that convergence is checked only on the system's
# initial states, iff the property requests it and the model declares an
# explicit (non-`AllStates`) initial-state set.
function apply_initial_restriction(base::TerminationCriteria, prop::Property, mp)
    restrict_to_initial(prop) || return base
    init = initial_states(mp)
    return init isa AllStates ? base : InitialStateCriteria(base, init)
end

# Default model checking algorithm termination: residual/iteration-based value
# iteration. `GeneralizedSamplingbasedRobustDynamicProgramming` overrides this
# with a gap-based base criterion (see `gsrdp.jl`).
function termination_criteria(::ModelCheckingAlgorithm, spec::Specification, mp)
    prop = system_property(spec)
    return apply_initial_restriction(base_termination_criteria(prop), prop, mp)
end
