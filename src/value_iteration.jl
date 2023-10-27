abstract type TerminationCriteria end

struct FixedIterationsCriteria <: TerminationCriteria
    n::Int
end
(f::FixedIterationsCriteria)(k, prev_V, V) = k >= f.n

struct CovergenceCriteria <: TerminationCriteria
    tol::Float64
end
function (f::CovergenceCriteria)(k, prev_V, V)
    for j in eachindex(V)
        if abs(prev_V[j] - V[j]) > f.tol
            return false
        end
    end

    return true
end

function interval_value_iteration(problem::Problem{<:IntervalMarkovChain, <:AbstractReachability}, termination_criteria::TerminationCriteria; max = true, discount = 1.0)
    mc = system(problem)
    spec = specification(problem)

    prob = transition_prob(mc)
    terminal = terminal_states(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    prev_V = construct_value_function(p)
    prev_V[terminal] .= 1.0

    V = similar(prev_V)
    V[terminal] .= prev_V[terminal]

    nonterminal = setdiff(collect(1:length(V)), terminal)

    step!(ordering, p, prob, prev_V, V, nonterminal; max = max, discount = discount)
    k = 1

    while !termination_criteria(k, prev_V, V)
        copyto!(prev_V, V)
        step!(ordering, p, prob, prev_V, V, nonterminal; max = max, discount = discount)
        k += 1
    end

    # Reuse prev_V to store the latest difference
    prev_V .-= V

    return V, k, prev_V
end

function construct_value_function(p::Vector{<:StateIntervalProbabilities{R}}) where {R}
    V = zeros(R, length(p))
    return V
end

function construct_value_function(p::MatrixIntervalProbabilities{R}) where {R}
    V = zeros(R, size(p, 1))
    return V
end

function step!(
    ordering,
    p,
    prob::Vector{<:StateIntervalProbabilities},
    prev_V,
    V,
    indices;
    max,
    discount
)
    partial_ominmax!(ordering, p, prob, V, indices; max = max)

    @inbounds for j in indices
        V[j] = discount * dot(p[j], prev_V)
    end
end

function step!(ordering, p, prob::MatrixIntervalProbabilities, prev_V, V, indices; max, discount)
    partial_ominmax!(ordering, p, prob, V, indices; max = max)

    @inbounds for j in indices
        V[j] = discount .* dot(view(p, :, j), prev_V)
    end
end
