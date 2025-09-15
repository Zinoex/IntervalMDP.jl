
struct FactoredRobustMarkovDecisionProcess{
    N,
    M,
    P <: NTuple{N, Marginal},
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
        initial_states::VI,
        check::Val{true},
    ) where {N, M, P <: NTuple{N, Marginal}, VI <: InitialStates}
        check_rmdp(state_vars, action_vars, source_dims, transition, initial_states)

        return new{N, M, P, VI}(state_vars, action_vars, source_dims, transition, initial_states)
    end

    function FactoredRobustMarkovDecisionProcess(
        state_vars::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        transition::P,
        initial_states::VI,
        check::Val{false},
    ) where {N, M, P <: NTuple{N, Marginal}, VI <: InitialStates}
        return new{N, M, P, VI}(state_vars, action_vars, source_dims, transition, initial_states)
    end
end
const FactoredRMDP = FactoredRobustMarkovDecisionProcess

function FactoredRMDP(
    state_vars::NTuple{N, Int32},
    action_vars::NTuple{M, Int32},
    source_dims::NTuple{N, Int32},
    transition::P,
    initial_states::VI = AllStates(),
) where {N, M, P <: NTuple{N, Marginal}, VI <: InitialStates}
    return FactoredRobustMarkovDecisionProcess(state_vars, action_vars, source_dims, transition, initial_states, Val(true))
end

function FactoredRMDP(
    state_vars::NTuple{N, <:Integer},
    action_vars::NTuple{M, <:Integer},
    source_dims::NTuple{N, <:Integer},
    transition::NTuple{N, Marginal},
    initial_states::VI = AllStates(),
) where {N, M, VI <: InitialStates}
    state_vars_32 = Int32.(state_vars)
    action_vars_32 = Int32.(action_vars)
    source_dims_32 = Int32.(source_dims)

    return FactoredRobustMarkovDecisionProcess(state_vars_32, action_vars_32, source_dims_32, transition, initial_states)
end

function FactoredRMDP(
    state_vars::NTuple{N, <:Integer},
    action_vars::NTuple{M, <:Integer},
    transition::NTuple{N, Marginal},
    initial_states::VI = AllStates(),
) where {N, M, VI <: InitialStates}
    return FactoredRobustMarkovDecisionProcess(state_vars, action_vars, state_vars, transition, initial_states)
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

        expected_source_shape = getindex.((source_dims,), state_variables(marginal))
        if source_shape(marginal) != expected_source_shape
            throw(DimensionMismatch("Marginal $i has incorrect source shape. Expected $expected_source_shape, got $(source_shape(marginal))."))
        end

        expected_action_shape = getindex.((action_dims,), action_variables(marginal))
        if action_shape(marginal) != expected_action_shape
            throw(DimensionMismatch("Marginal $i has incorrect action shape. Expected $expected_action_shape, got $(action_shape(marginal))."))
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

state_variables(mdp::FactoredRMDP) = mdp.state_vars
state_variables(mdp::FactoredRMDP, r) = mdp.state_vars[r]
action_variables(mdp::FactoredRMDP) = mdp.action_vars
num_states(mdp::FactoredRMDP) = prod(state_variables(mdp))
num_actions(mdp::FactoredRMDP) = prod(action_variables(mdp))
marginals(mdp::FactoredRMDP) = mdp.transition
initial_states(mdp::FactoredRMDP) = mdp.initial_states

source_shape(m::FactoredRMDP) = m.source_dims
action_shape(m::FactoredRMDP) = m.action_vars

function Base.getindex(mdp::FactoredRMDP, r)
    return mdp.transition[r]
end

### Model type analysis
abstract type ModelType end

abstract type NonFactored <: ModelType end
struct IsIMDP <: NonFactored end # Interval MDP
struct IsRMDP <: NonFactored end # Robust MDP

abstract type Factored <: ModelType end
struct IsFIMDP <: ModelType end # Factored Interval MDP
struct IsFPMDP <: ModelType end # Factored Polytopic MDP
struct IsFRMDP <: ModelType end # Factored Robust MDP

# Single marginal - special case
modeltype(mdp::FactoredRMDP{1}) = modeltype(mdp, isinterval(mdp.transition[1]))
modeltype(::FactoredRMDP{1}, ::IsInterval) = IsIMDP()
modeltype(::FactoredRMDP{1}, ::IsNotInterval) = IsRMDP()

# General factored case

# Check if all marginals are interval ambiguity sets
modeltype(mdp::FactoredRMDP{N}) where {N} = modeltype(mdp, isinterval.(mdp.transition))
modeltype(::FactoredRMDP{N}, ::NTuple{N, IsInterval}) where {N} = IsFIMDP()

# If not, check if all marginals are polytopic ambiguity sets
modeltype(::FactoredRMDP{N}, ::NTuple{N, AbstractIsInterval}) where {N} = modeltype(mdp, ispolytopic.(mdp.transition))
modeltype(::FactoredRMDP{N}, ::NTuple{N, IsPolytopic}) where {N} = IsFPMDP()

# Otherwise, it is a general factored robust MDP
modeltype(::FactoredRMDP{N}, ::NTuple{N, AbstractIsPolytopic}) where {N} = IsFRMDP()


### Pretty printing
function Base.show(io::IO, mime::MIME"text/plain", mdp::FactoredRMDP)
    showsystem(io, "", "", mdp)
end

function showsystem(io::IO, first_prefix, prefix, mdp::FactoredRMDP{N, M}) where {N, M}
    println(io, first_prefix, styled"{code:FactoredRobustMarkovDecisionProcess}")
    println(io, prefix, "├─ ", N, styled" state variables with cardinality: {magenta:$(state_variables(mdp))}")
    println(io, prefix, "├─ ", M, styled" action variables with cardinality: {magenta:$(action_variables(mdp))}")
    if initial_states(mdp) isa AllStates
        println(io, prefix, "├─ ", styled"Initial states: {magenta:All states}")
    else
        println(io, prefix, "├─ ", styled"Initial states: {magenta:$(initial_states(mdp))}")
    end

    println(io, prefix, "├─ ", styled"Transition marginals:")
    marginal_prefix = prefix * "│  "
    for (i, marginal) in enumerate(mdp.transition[1:end - 1])
        println(io, marginal_prefix, "├─ Marginal $i: ")
        showmarginal(io, marginal_prefix * "│  ", marginal)
    end
    println(io, marginal_prefix, "└─ Marginal $(length(mdp.transition)): ")
    showmarginal(io, marginal_prefix * "   ", mdp.transition[end])

    showinferred(io, prefix, mdp)
end

function showinferred(io::IO, prefix, mdp::FactoredRMDP)
    println(io, prefix, "└─", styled"{red:Inferred properties}")
    prefix = prefix * "   "
    showmodeltype(io, prefix, mdp)
    println(io, prefix, "├─", styled"Number of states: {green:$(num_states(mdp))}")
    println(io, prefix, "├─", styled"Number of actions: {green:$(num_actions(mdp))}")

    default_alg = default_algorithm(mdp)
    showmcalgorithm(io, prefix, default_alg)
    showbellmanalg(io, prefix, modeltype(mdp), bellman_algorithm(default_alg))
end

showmodeltype(io::IO, prefix, mdp::FactoredRMDP) = showmodeltype(io, prefix, modeltype(mdp))

function showmodeltype(io::IO, prefix, ::IsFIMDP)
    println(io, prefix, "├─", styled"Model type: {green:Factored Interval MDP}")
end

function showmodeltype(io::IO, prefix, ::IsFPMDP)
    println(io, prefix, "├─", styled"Model type: {green:Factored Polytopic MDP}")
end

function showmodeltype(io::IO, prefix, ::IsFRMDP)
    println(io, prefix, "├─", styled"Model type: {green:Factored Robust MDP}")
end

function showmodeltype(io::IO, prefix, ::IsIMDP)
    println(io, prefix, "├─", styled"Model type: {green:Interval MDP}")
end

function showmodeltype(io::IO, prefix, ::IsRMDP)
    println(io, prefix, "├─", styled"Model type: {green:Robust MDP}")
end