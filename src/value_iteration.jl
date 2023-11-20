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

function interval_value_iteration(
    problem::Problem{<:IntervalMarkovChain, <:AbstractReachability};
    upper_bound = true,
    discount = 1.0,
)
    mc = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(problem)

    prob = transition_prob(mc)
    terminal = terminal_states(spec)
    target = reach(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    prev_V = construct_value_function(p, num_states(mc))
    view(prev_V, target) .= 1.0

    V = copy(prev_V)

    nonterminal = construct_nonterminal(mc, terminal)
    # Important to use view to avoid copying
    V_nonterminal = reshape(view(V, nonterminal), 1, length(nonterminal))

    step_imc!(
        ordering,
        p,
        prob,
        prev_V,
        V_nonterminal,
        nonterminal;
        upper_bound = upper_bound,
        discount = discount,
    )
    prev_V .-= V
    rmul!(prev_V, -1.0)
    k = 1

    while !term_criteria(V, k, prev_V)
        copyto!(prev_V, V)
        step_imc!(
            ordering,
            p,
            prob,
            prev_V,
            V_nonterminal,
            nonterminal;
            upper_bound = upper_bound,
            discount = discount,
        )

        # Reuse prev_V to store the latest difference
        prev_V .-= V
        rmul!(prev_V, -1.0)
        k += 1
    end

    return V, k, prev_V
end

function construct_value_function(
    ::AbstractVector{<:AbstractVector{R}},
    num_states,
) where {R}
    V = zeros(R, num_states)
    return V
end

function construct_value_function(::AbstractMatrix{R}, num_states) where {R}
    V = zeros(R, num_states)
    return V
end

function construct_nonterminal(mc::IntervalMarkovChain, terminal)
    return setdiff(collect(1:num_states(mc)), terminal)
end

function step_imc!(
    ordering,
    p,
    prob::Vector{<:StateIntervalProbabilities},
    prev_V,
    V,
    indices;
    upper_bound,
    discount,
)
    partial_ominmax!(ordering, p, prob, prev_V, indices; max = upper_bound)

    @inbounds for (i, j) in enumerate(indices)
        V[i] = discount * dot(p[j], prev_V)
    end

    return V
end

function step_imc!(
    ordering,
    p,
    prob::MatrixIntervalProbabilities,
    prev_V,
    V,
    indices;
    upper_bound,
    discount,
)
    partial_ominmax!(ordering, p, prob, prev_V, indices; max = upper_bound)

    p = view(p, :, indices)
    mul!(V, transpose(prev_V), p)
    rmul!(V, discount)

    return V
end

function interval_value_iteration(
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
    terminal = terminal_states(spec)
    target = reach(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    prev_V = construct_value_function(p, num_states(mdp))
    view(prev_V, target) .= 1.0

    V = copy(prev_V)

    nonterminal, nonterminal_actions = construct_nonterminal(mdp, terminal)
    V_nonterminal = similar(V, 1, length(nonterminal_actions))

    step_imdp!(
        ordering,
        p,
        prob,
        sptr,
        maxactions,
        prev_V,
        V,
        V_nonterminal,
        nonterminal,
        nonterminal_actions;
        maximize = maximize,
        upper_bound = upper_bound,
        discount = discount,
    )
    prev_V .-= V
    rmul!(prev_V, -1.0)
    k = 1

    while !term_criteria(V, k, prev_V)
        copyto!(prev_V, V)
        step_imdp!(
            ordering,
            p,
            prob,
            sptr,
            maxactions,
            prev_V,
            V,
            V_nonterminal,
            nonterminal,
            nonterminal_actions;
            maximize = maximize,
            upper_bound = upper_bound,
            discount = discount,
        )

        # Reuse prev_V to store the latest difference
        prev_V .-= V
        rmul!(prev_V, -1.0)
        k += 1
    end

    return V, k, prev_V
end

function construct_nonterminal(mdp::IntervalMarkovDecisionProcess, terminal)
    sptr = stateptr(mdp)

    nonterminal = setdiff(collect(1:num_states(mdp)), terminal)
    nonterminal_actions =
        mapreduce(i -> collect(sptr[i]:(sptr[i + 1] - 1)), vcat, nonterminal)
    return nonterminal, nonterminal_actions
end

function step_imdp!(
    ordering,
    p,
    prob::Vector{<:StateIntervalProbabilities},
    stateptr,
    maxactions,
    prev_V,
    V,
    V_nonterminal,
    state_indices,
    action_indices;
    maximize,
    upper_bound,
    discount,
)
    partial_ominmax!(ordering, p, prob, prev_V, action_indices; max = upper_bound)

    optfun = maximize ? max : min

    @inbounds for j in action_indices
        @inbounds V_nonterminal[j] = discount * dot(p[j], prev_V)
    end

    @inbounds for j in state_indices
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        @inbounds V[j] = optfun(view(V_nonterminal, s1:s2 - 1))
    end

    return V
end

function step_imdp!(
    ordering,
    p,
    prob::MatrixIntervalProbabilities,
    stateptr,
    maxactions,
    prev_V,
    V,
    V_nonterminal,
    state_indices,
    action_indices;
    maximize,
    upper_bound,
    discount,
)
    partial_ominmax!(ordering, p, prob, prev_V, action_indices; max = upper_bound)

    optfun = maximize ? maximum : minimum

    p = view(p, :, action_indices)
    mul!(V_nonterminal, transpose(prev_V), p)
    rmul!(V_nonterminal, discount)

    @inbounds for j in state_indices
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        @inbounds V[j] = optfun(view(V_nonterminal, s1:s2 - 1))
    end

    return V
end
