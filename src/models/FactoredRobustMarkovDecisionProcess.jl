
struct FactoredRobustMarkovDecisionProcess{
    N,
    M,
    P <: NTuple{N, <:AbstractMarginal},
    VI <: InitialStates,
} <: IntervalMarkovProcess
    state_vars::NTuple{N, Int32}   # N is the number of state variables and state_vars[n] is the number of states for state variable n
    action_vars::NTuple{M, Int32}  # M is the number of action variables and action_vars[m] is the number of actions for action variable m

    source_dims::NTuple{N, Int32} 
    
    transition::P
    initial_states::VI

    function FactoredRobustMarkovDecisionProcess(
        state_vars::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        transition::P,
        initial_states::VI = nothing,
    ) where {N, M, P <: NTuple{N, <:AbstractMarginal}, VI <: InitialStates}
        check_rmdp(state_vars, action_vars, source_dims, transition, initial_states)

        return new{N, M, P, VI}(state_vars, action_vars, source_dims, transition, initial_states)
    end
end
const FactoredRMDP = FactoredRobustMarkovDecisionProcess

function FactoredRMDP(
    state_vars::NTuple{N, Int},
    action_vars::NTuple{M, Int},
    source_dims::NTuple{N, Int},
    transition::P,
    initial_states::VI = AllStates(),
) where {N, M, P <: NTuple{N, <:AbstractMarginal}, VI <: InitialStates}
    state_vars_32 = Int32.(state_vars)
    action_vars_32 = Int32.(action_vars)
    source_dims_32 = Int32.(source_dims)

    return FactoredRobustMarkovDecisionProcess(state_vars_32, action_vars_32, source_dims_32, transition, initial_states)
end

function FactoredRMDP(
    state_vars::NTuple{N, <:Integer},
    action_vars::NTuple{M, <:Integer},
    transition::P,
    initial_states::VI = AllStates(),
) where {N, M, P <: NTuple{N, <:AbstractMarginal}, VI <: InitialStates}
    return FactoredRobustMarkovDecisionProcess{N, M, P, VI}(state_vars, action_vars, state_vars, transition, initial_states)
end

function check_rmdp(state_vars, action_vars, source_dims, transition, initial_states)
    check_state_variables(state_vars, source_dims)
    check_action_variables(action_vars)
    check_transition(state_vars, action_vars, source_dims, transition)
    check_initial_states(state_vars, initial_states)
end

function check_state_variables(state_vars, source_dims)
    if any(n -> n <= 0, state_vars)
        throw(ArgumentError("All state variables must be positive integers."))
    end

    if any(i -> source_dims[i] <= 0 || source_dims[i] > state_vars[i], eachindex(state_vars))
        throw(ArgumentError("All source dimensions must be positive integers and less than or equal to the corresponding state variable."))
    end
end

function check_action_variables(action_vars)
    if any(x -> x <= 0, action_vars)
        throw(ArgumentError("All action variables must be positive integers."))
    end
end

function check_transition(state_dims, action_dims, source_dims, transition)
    for (i, marginal) in enumerate(transition)
        if num_target(marginal) != state_dims[i]
            throw(DimensionMismatch("Marginal $i has incorrect number of target states. Expected $(state_dims[i]), got $(num_target(marginal))."))
        end

        if source_shape(marginal) != getindex.(Tuple(source_dims), state_variables(marginal))   # source_dims[state_variables(marginal)]
            throw(DimensionMismatch("Marginal $i has incorrect source shape. Expected $source_dims, got $(source_shape(marginal))."))
        end

        if action_shape(marginal) != getindex.(Tuple(action_dims), action_variables(marginal))
            throw(DimensionMismatch("Marginal $i has incorrect action shape. Expected $action_dims, got $(action_shape(marginal))."))
        end
    end
end

function check_initial_states(state_vars, initial_states)
    if initial_states isa AllStates
        return
    end

    N = length(state_vars)
    for initial_state in initial_states
        if length(initial_state) != N
            throw(DimensionMismatch("Each initial state must have length $N."))
        end

        if !all(1 .<= initial_state .<= state_vars)
            throw(DimensionMismatch("Each initial state must be within the valid range of states (should be 1 .<= initial_state <= $state_vars, was initial_state=$initial_state)."))
        end
    end
end

state_variables(rmdp::FactoredRobustMarkovDecisionProcess) = rmdp.state_vars
action_variables(rmdp::FactoredRobustMarkovDecisionProcess) = rmdp.action_vars
num_states(rmdp::FactoredRobustMarkovDecisionProcess) = prod(state_variables(rmdp))
num_actions(rmdp::FactoredRobustMarkovDecisionProcess) = prod(action_variables(rmdp))
marginals(rmdp::FactoredRobustMarkovDecisionProcess) = rmdp.transition
initial_states(rmdp::FactoredRobustMarkovDecisionProcess) = rmdp.initial_states

source_shape(m::FactoredRobustMarkovDecisionProcess) = m.source_dims
action_shape(m::FactoredRobustMarkovDecisionProcess) = m.action_vars

function Base.getindex(rmdp::FactoredRMDP, r)
    return rmdp.transition[r]
end