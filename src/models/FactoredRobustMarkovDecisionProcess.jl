"""
    FactoredRobustMarkovDecisionProcess{N, M, P <: NTuple{N, Marginal}, VI <: InitialStates} <: IntervalMarkovProcess

Factored Robust Markov Decision Processes (fRMDPs) [schnitzer2025efficient, delgado2011efficient](@cite) are
an extension of Robust Markov Decision Processes (RMDPs) [nilim2005robust, wiesemann2013robust, suilen2024robust](@cite)
that incorporate a factored representation of the state and action spaces, i.e. with state and action variables.

Formally, a fRMDP ``M`` is a tuple ``M = (S, S_0, A, \\mathcal{G}, \\Gamma)``, where

- ``S = S_1 \\times \\cdots \\times S_n`` is a finite set of joint states with ``S_i`` 
    being a finite set of states for the ``i``-th state variable,
- ``S_0 \\subseteq S`` is a set of initial states,
- ``A = A_1 \\times \\cdots \\times A_m`` is a finite set of joint actions with ``A_j`` 
    being a finite set of actions for the ``j``-th action variable,
- ``\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})`` is a directed bipartite graph with nodes 
    ``\\mathcal{V} = \\mathcal{V}_{ind} \\cup \\mathcal{V}_{cond} = \\{S_1, \\ldots, S_n, A_1, \\ldots, A_m\\} \\cup \\{S'_1, \\ldots, S'_n\\}``
    representing the state and action variables and their next-state counterparts, and edges 
    ``\\mathcal{E} \\subseteq \\mathcal{V}_{ind} \\times \\mathcal{V}_{cond}``
    representing dependencies of ``S'_i`` on ``S_j`` and ``A_k``,
- ``\\Gamma = \\{\\Gamma_{s, a}\\}_{s \\in S, a \\in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, 
    where each ``\\Gamma_{s, a} = \\bigotimes_{i=1}^n \\Gamma^i_{\\text{Pa}_\\mathcal{G}(S'_i) \\cap (s, a)}`` is
    a product of ambiguity sets ``\\Gamma^i_{\\text{Pa}_\\mathcal{G}(S'_i) \\cap (s, a)}`` along each marginal ``i`` conditional
    on the values in ``(s, a)`` of the parent variables ``\\text{Pa}_\\mathcal{G}(S'_i)`` of ``S'_i`` in ``\\mathcal{G}``, i.e.
```math
    \\Gamma_{s, a} = \\left\\{ \\gamma \\in \\mathcal{D}(S) \\,:\\, \\gamma(t) = \\prod_{i=1}^n \\gamma^i(t_i | s_{\\text{Pa}_{\\mathcal{G}_S}(S'_i)}, a_{\\text{Pa}_{\\mathcal{G}_A}(S'_i)}), \\, \\gamma^i(\\cdot | s_{\\text{Pa}_{\\mathcal{G}_S}(S'_i)}, a_{\\text{Pa}_{\\mathcal{G}_A}(S'_i)}) \\in \\Gamma^i_{\\text{Pa}_\\mathcal{G}(S'_i)} \\right\\}.
```

For a given source-action pair ``(s, a) \\in S \\times A``, any distribution ``\\gamma_{s, a} \\in \\Gamma_{s, a}`` is called a feasible distribution,
and feasible transitions are triplets ``(s, a, t) \\in S \\times A \\times S`` where ``t \\in \\mathop{supp}(\\gamma_{s, a})`` for any feasible distribution ``\\gamma_{s, a} \\in \\Gamma_{s, a}``.

### Type parameters
- `N` is the number of state variables.
- `M` is the number of action variables.
- `P <: NTuple{N, Marginal}` is a tuple type with a (potentially different) type for each marginal.
- `VI <: InitialStates` is the type of initial states.

### Fields
- `state_vars::NTuple{N, Int32}`: the number of values ``|S_i|`` for each state variable ``S_i`` as a tuple.
- `action_vars::NTuple{M, Int32}`: the number of values ``|A_k|`` for each action variable ``A_k`` as a tuple.
- `source_dims::NTuple{N, Int32}`: for systems with terminal states along certain slices, it is possible to avoid
    specifying them by using `source_dims` less than `state_vars`; this is useful e.g. in building abstractions.
    The terminal states must be the last value for the slice dimension. If not supplied, it is assumed `source_dims == state_vars`.
- `transition::P` is the marginal ambiguity sets. For a given source-action pair ``(s, a) \\in S \\times A``,
    any [`Marginal`](@ref) element of `transition` subselects `s` and `a` corresponding to its [`state_variables`](@ref)
    and [`action_variables`](@ref), i.e. it encodes the operation `\\text{Pa}_\\mathcal{G}(S'_i) \\cap (s, a)`.
    The underlying `ambiguity_sets` object on `Marginal` encodes ``\\Gamma^i_{\\text{Pa}_\\mathcal{G}(S'_i) \\cap (s, a)}``
    for all values of ``\\text{Pa}_\\mathcal{G}(S'_i)``. See [`Marginal`](@ref) for details about the layout of the underlying
    `AbstractAmbiguitySets` object.
- `initial_states::VI`: stores a representation of `S_0`. If no set of initial_states are given, then it is simply assigned
    the zero-byte object `AllStates()`, which represents that all states are potential initial states. It is not used within
    the value iteration.

### Example
```jldoctest
using IntervalMDP

state_vars = (2, 3)
action_vars = (1, 2)

state_indices = (1, 2)
action_indices = (1,)
state_dims = (2, 3)
action_dims = (1,)
marginal1 = Marginal(IntervalAmbiguitySets(;
    # 6 ambiguity sets = 2 * 3 source states, 1 action
    # Column layout: (a¹₁, s¹₁, s²₁), (a¹₁, s¹₂, s²₁), (a¹₁, s¹₁, s²₂), (a¹₁, s¹₂, s²₂), (a¹₁, s¹₁, s²₃), (a¹₁, s¹₂, s²₃)
    # Equivalent to CartesianIndices(actions_dims..., state_dims...), i.e. actions first, then states in lexicographic order
    lower = [
        1/15  7/30  1/15  13/30  4/15  1/6
        2/5   7/30  1/30  11/30  2/15  1/10
    ],
    upper = [
        17/30  7/10   2/3   4/5  7/10  2/3
        9/10   13/15  9/10  5/6  4/5   14/15
    ]
), state_indices, action_indices, state_dims, action_dims)

state_indices = (2,)
action_indices = (2,)
state_dims = (3,)
action_dims = (2,)
marginal2 = Marginal(IntervalAmbiguitySets(;
    # 6 ambiguity sets = 3 source states, 2 actions
    # Column layout: (a²₁, s²₁), (a²₂, s²₁), (a²₁, s²₂), (a²₂, s²₂), (a²₁, s²₃), (a²₂, s²₃)
    # Equivalent to CartesianIndices(actions_dims..., state_dims...), i.e. actions first, then states in lexicographic order
    lower = [
        1/30  1/3   1/6   1/15  2/5   2/15
        4/15  1/4   1/6   1/30  2/15  1/30
        2/15  7/30  1/10  7/30  7/15  1/5
    ],
    upper = [
        2/3    7/15  4/5    11/30  19/30  1/2
        23/30  4/5   23/30  3/5    7/10   8/15
        7/15   4/5   23/30  7/10   7/15   23/30
    ]
), state_indices, action_indices, state_dims, action_dims)

initial_states = [(1, 1)]  # Initial states are optional
mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2), initial_states)

# output

FactoredRobustMarkovDecisionProcess
├─ 2 state variables with cardinality: (2, 3)
├─ 2 action variables with cardinality: (1, 2)
├─ Initial states: [(1, 1)]
├─ Transition marginals:
│  ├─ Marginal 1:
│  │  ├─ Conditional variables: states = (1, 2), actions = (1,)
│  │  └─ Ambiguity set type: Interval (dense, Matrix{Float64})
│  └─ Marginal 2:
│     ├─ Conditional variables: states = (2,), actions = (2,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Factored Interval MDP
   ├─Number of states: 6
   ├─Number of actions: 2
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: Binary tree LP McCormick Relaxation
```
"""
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

        return new{N, M, P, VI}(
            state_vars,
            action_vars,
            source_dims,
            transition,
            initial_states,
        )
    end

    function FactoredRobustMarkovDecisionProcess(
        state_vars::NTuple{N, Int32},
        action_vars::NTuple{M, Int32},
        source_dims::NTuple{N, Int32},
        transition::P,
        initial_states::VI,
        check::Val{false},
    ) where {N, M, P <: NTuple{N, Marginal}, VI <: InitialStates}
        return new{N, M, P, VI}(
            state_vars,
            action_vars,
            source_dims,
            transition,
            initial_states,
        )
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
    return FactoredRobustMarkovDecisionProcess(
        state_vars,
        action_vars,
        source_dims,
        transition,
        initial_states,
        Val(true),
    )
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

    return FactoredRobustMarkovDecisionProcess(
        state_vars_32,
        action_vars_32,
        source_dims_32,
        transition,
        initial_states,
    )
end

function FactoredRMDP(
    state_vars::NTuple{N, <:Integer},
    action_vars::NTuple{M, <:Integer},
    transition::NTuple{N, Marginal},
    initial_states::VI = AllStates(),
) where {N, M, VI <: InitialStates}
    return FactoredRobustMarkovDecisionProcess(
        state_vars,
        action_vars,
        state_vars,
        transition,
        initial_states,
    )
end

function check_rmdp(state_vars, action_vars, source_dims, transition, initial_states)
    check_state_values(state_vars, source_dims)
    check_action_values(action_vars)
    check_transition(state_vars, action_vars, source_dims, transition)
    check_initial_states(state_vars, initial_states)
end

function check_state_values(state_vars, source_dims)
    if any(n -> n <= 0, state_vars)
        throw(ArgumentError("All state variables must be positive integers."))
    end

    if any(
        i -> source_dims[i] <= 0 || source_dims[i] > state_vars[i],
        eachindex(state_vars),
    )
        throw(
            ArgumentError(
                "All source dimensions must be positive integers and less than or equal to the corresponding state variable.",
            ),
        )
    end
end

function check_action_values(action_vars)
    if any(x -> x <= 0, action_vars)
        throw(ArgumentError("All action variables must be positive integers."))
    end
end

function check_transition(state_dims, action_dims, source_dims, transition)
    for (i, marginal) in enumerate(transition)
        if num_target(marginal) != state_dims[i]
            throw(
                DimensionMismatch(
                    "Marginal $i has incorrect number of target states. Expected $(state_dims[i]), got $(num_target(marginal)).",
                ),
            )
        end

        expected_source_shape = getindex.((source_dims,), state_variables(marginal))
        if source_shape(marginal) != expected_source_shape
            throw(
                DimensionMismatch(
                    "Marginal $i has incorrect source shape. Expected $expected_source_shape, got $(source_shape(marginal)).",
                ),
            )
        end

        expected_action_shape = getindex.((action_dims,), action_variables(marginal))
        if action_shape(marginal) != expected_action_shape
            throw(
                DimensionMismatch(
                    "Marginal $i has incorrect action shape. Expected $expected_action_shape, got $(action_shape(marginal)).",
                ),
            )
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

        if !all(1 .<= Tuple(initial_state) .<= state_vars)
            throw(
                DimensionMismatch(
                    "Each initial state must be within the valid range of states (should be 1 .<= initial_state <= $state_vars, was initial_state=$initial_state).",
                ),
            )
        end
    end
end

"""
    state_values(mdp::FactoredRMDP)
    
Return a tuple with the number of states for each state variable in the fRMDP.
"""
state_values(mdp::FactoredRMDP) = mdp.state_vars
state_values(mdp::FactoredRMDP, r) = mdp.state_vars[r]

"""
    action_values(mdp::FactoredRMDP)

Return a tuple with the number of actions for each action variable in the fRMDP.
"""
action_values(mdp::FactoredRMDP) = mdp.action_vars

"""
    marginals(mdp::FactoredRMDP)

Return the marginals of the fRMDP.
"""
marginals(mdp::FactoredRMDP) = mdp.transition

num_states(mdp::FactoredRMDP) = prod(state_values(mdp))
num_actions(mdp::FactoredRMDP) = prod(action_values(mdp))
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
modeltype(::FactoredRMDP{N}, ::NTuple{N, AbstractIsInterval}) where {N} = IsFRMDP()

### Pretty printing
function Base.show(io::IO, mime::MIME"text/plain", mdp::FactoredRMDP)
    showsystem(io, "", "", mdp)
end

function showsystem(io::IO, first_prefix, prefix, mdp::FactoredRMDP{N, M}) where {N, M}
    println(io, first_prefix, styled"{code:FactoredRobustMarkovDecisionProcess}")
    println(
        io,
        prefix,
        "├─ ",
        N,
        styled" state variables with cardinality: {magenta:$(state_values(mdp))}",
    )
    println(
        io,
        prefix,
        "├─ ",
        M,
        styled" action variables with cardinality: {magenta:$(action_values(mdp))}",
    )
    if initial_states(mdp) isa AllStates
        println(io, prefix, "├─ ", styled"Initial states: {magenta:All states}")
    else
        println(io, prefix, "├─ ", styled"Initial states: {magenta:$(initial_states(mdp))}")
    end

    println(io, prefix, "├─ ", styled"Transition marginals:")
    marginal_prefix = prefix * "│  "
    for (i, marginal) in enumerate(mdp.transition[1:(end - 1)])
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

function showmodeltype(io::IO, prefix, ::IsFRMDP)
    println(io, prefix, "├─", styled"Model type: {green:Factored Robust MDP}")
end

function showmodeltype(io::IO, prefix, ::IsIMDP)
    println(io, prefix, "├─", styled"Model type: {green:Interval MDP}")
end

function showmodeltype(io::IO, prefix, ::IsRMDP)
    println(io, prefix, "├─", styled"Model type: {green:Robust MDP}")
end
