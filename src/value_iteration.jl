abstract type TerminationCriteria end
termination_criteria(problem::Problem) = termination_criteria(specification(problem))

struct FixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::FixedIterationsCriteria)(V, k, u) = k >= f.n
termination_criteria(spec::Union{FiniteTimeReachability, FiniteTimeReachAvoid}) =
    FixedIterationsCriteria(time_horizon(spec))

struct CovergenceCriteria{T <: AbstractFloat} <: TerminationCriteria
    tol::T
end
(f::CovergenceCriteria)(V, k, u) = maximum(u) < f.tol
termination_criteria(spec::Union{InfiniteTimeReachability, InfiniteTimeReachAvoid}) =
    CovergenceCriteria(eps(spec))

"""
    value_iteration(problem::Problem{<:IntervalMarkovChain, <:AbstractReachability};
        upper_bound = true,
        discount = 1.0
    )

Solve reachability and reach-avoid problems using value iteration for interval Markov chain. If `upper_bound == true`
then the optimistic probability of reachability or reach-avoid is computed. Otherwise, the pessimistic probability is computed.

### Examples

```jldoctest
prob = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.0
        0.1 0.3 0.0
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0
        0.6 0.5 0.0
        0.7 0.3 1.0
    ],
)

mc = IntervalMarkovChain(prob, 1)

terminal_states = [3]
time_horizon = 10
problem = Problem(mc, FiniteTimeReachability(terminal_states, time_horizon))
V, k, residual = value_iteration(problem; upper_bound = false)
```

"""
function value_iteration(
    problem::Problem{<:IntervalMarkovChain, <:AbstractReachability};
    upper_bound = true,
    discount = 1.0,
)
    # TODO: The discount factor probably ought to be a parameter on a reward min/max specification.
    mc = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(problem)

    prob = transition_prob(mc)
    target = reach(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    value_function = IMCValueFunction(problem)
    initialize!(value_function, target, 1.0)

    step_imc!(
        ordering,
        p,
        prob,
        value_function;
        upper_bound = upper_bound,
        discount = discount,
    )
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)
        step_imc!(
            ordering,
            p,
            prob,
            value_function;
            upper_bound = upper_bound,
            discount = discount,
        )

        k += 1
    end

    # lastdiff! uses prev to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.cur, k, value_function.prev
end

function construct_value_function(::AbstractMatrix{R}, num_states) where {R}
    V = zeros(R, num_states)
    return V
end

function construct_nonterminal(mc::IntervalMarkovChain, terminal)
    return setdiff(collect(1:num_states(mc)), terminal)
end

struct IMCValueFunction
    prev::Any
    prev_transpose::Any
    cur::Any
    nonterminal::Any
    nonterminal_indices::Any
end

function IMCValueFunction(problem::P) where {P <: Problem{<:IntervalMarkovChain}}
    mc = system(problem)
    spec = specification(problem)
    terminal = terminal_states(spec)

    prev = construct_value_function(gap(transition_prob(mc)), num_states(mc))
    prev_transpose = transpose(prev)
    cur = copy(prev)

    nonterminal_indices = construct_nonterminal(mc, terminal)
    # Important to use view to avoid copying
    nonterminal = reshape(view(cur, nonterminal_indices), 1, length(nonterminal_indices))

    return IMCValueFunction(prev, prev_transpose, cur, nonterminal, nonterminal_indices)
end

function initialize!(V, indices, values)
    view(V.prev, indices) .= values
    view(V.cur, indices) .= values

    return V
end

function lastdiff!(V)
    # Reuse prev to store the latest difference
    V.prev .-= V.cur
    rmul!(V.prev, -1.0)

    return V.prev
end

function nextiteration!(V)
    copyto!(V.prev, V.cur)

    return V
end

function step_imc!(
    ordering,
    p,
    prob::IntervalProbabilities,
    value_function::IMCValueFunction;
    upper_bound,
    discount,
)
    indices = value_function.nonterminal_indices
    partial_ominmax!(ordering, p, prob, value_function.prev, indices; max = upper_bound)

    p = view(p, :, indices)
    mul!(value_function.nonterminal, value_function.prev_transpose, p)
    rmul!(value_function.nonterminal, discount)

    return value_function
end


"""
    value_iteration(problem::Problem{<:IntervalMarkovDecisionProcess, <:AbstractReachability};
        maximize = true,
        upper_bound = true,
        discount = 1.0
    )

Solve reachability and reach-avoid problems using value iteration for interval Markov decision processes. If `upper_bound == true`
then the optimistic probability of reachability or reach-avoid is computed. Otherwise, the pessimistic probability is computed.
The maximize keyword argument determines whether the action that maximizes or minimizes the probability is chosen.

### Examples

```jldoctest
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(;
    lower = [0.0; 0.0; 1.0],
    upper = [0.0; 0.0; 1.0]
)

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
initial_state = 1
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)

terminal_states = [3]
time_horizon = 10
problem = Problem(mdp, FiniteTimeReachability(terminal_states, time_horizon))
V, k, residual = value_iteration(problem; maximize = true, upper_bound = false)
```

"""
function value_iteration(
    problem::Problem{<:IntervalMarkovDecisionProcess, <:AbstractReachability};
    maximize = true,
    upper_bound = true,
    discount = 1.0,
)
    mdp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(problem)

    prob = transition_prob(mdp)
    sptr = stateptr(mdp)
    maxactions = maximum(diff(sptr))
    target = reach(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    value_function = IMDPValueFunction(problem)
    initialize!(value_function, target, 1.0)

    step_imdp!(
        ordering,
        p,
        prob,
        sptr,
        maxactions,
        value_function;
        maximize = maximize,
        upper_bound = upper_bound,
        discount = discount,
    )
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)
        step_imdp!(
            ordering,
            p,
            prob,
            sptr,
            maxactions,
            value_function;
            maximize = maximize,
            upper_bound = upper_bound,
            discount = discount,
        )

        k += 1
    end

    # lastdiff! uses prev to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.cur, k, value_function.prev
end

function construct_nonterminal(mdp::IntervalMarkovDecisionProcess, terminal)
    sptr = stateptr(mdp)

    nonterminal = setdiff(collect(1:num_states(mdp)), terminal)
    nonterminal_actions =
        mapreduce(i -> collect(sptr[i]:(sptr[i + 1] - 1)), vcat, nonterminal)
    return nonterminal, nonterminal_actions
end

struct IMDPValueFunction
    prev::Any
    prev_transpose::Any
    cur::Any
    nonterminal::Any
    nonterminal_states::Any
    nonterminal_actions::Any
end

function IMDPValueFunction(problem::P) where {P <: Problem{<:IntervalMarkovDecisionProcess}}
    mdp = system(problem)
    spec = specification(problem)
    terminal = terminal_states(spec)

    prev = construct_value_function(gap(transition_prob(mdp)), num_states(mdp))
    prev_transpose = transpose(prev)
    cur = copy(prev)

    nonterminal_states, nonterminal_actions = construct_nonterminal(mdp, terminal)
    nonterminal = similar(cur, 1, length(nonterminal_actions))

    return IMDPValueFunction(
        prev,
        prev_transpose,
        cur,
        nonterminal,
        nonterminal_states,
        nonterminal_actions,
    )
end

function step_imdp!(
    ordering,
    p,
    prob::IntervalProbabilities,
    stateptr,
    maxactions,
    value_function;
    maximize,
    upper_bound,
    discount,
)
    partial_ominmax!(
        ordering,
        p,
        prob,
        value_function.prev,
        value_function.nonterminal_actions;
        max = upper_bound,
    )

    optfun = maximize ? maximum : minimum

    p = view(p, :, value_function.nonterminal_actions)
    mul!(value_function.nonterminal, value_function.prev_transpose, p)
    rmul!(value_function.nonterminal, discount)

    @inbounds for j in value_function.nonterminal_states
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        @inbounds value_function.cur[j] =
            optfun(view(value_function.nonterminal, s1:(s2 - 1)))
    end

    return value_function
end
