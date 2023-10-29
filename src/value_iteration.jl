abstract type TerminationCriteria end

struct FixedIterationsCriteria <: TerminationCriteria
    n::Int
end
(f::FixedIterationsCriteria)(k, prev_V, V) = k >= f.n

struct CovergenceCriteria <: TerminationCriteria
    tol::Float64
end
(f::CovergenceCriteria)(k, prev_V, V) = maximum(abs.(prev_V - V)) < f.tol

function interval_value_iteration(prob, V, fix_indices, termination_criteria; max = true)
    # It is more efficient to use allocate first and reuse across iterations
    ordering = construct_ordering(gap(prob))
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    prev_V = copy(V)         # We need to start from the starting value
    V = similar(V)           # This becomes the output after each iteration

    indices = setdiff(collect(1:length(V)), fix_indices)
    V[fix_indices] .= prev_V[fix_indices]

    step!(ordering, p, prob, prev_V, V, indices; max = max)
    k = 1

    while !termination_criteria(k, prev_V, V)
        copyto!(prev_V, V)
        step!(ordering, p, prob, prev_V, V, indices; max = max)
        k += 1
    end

    return V, k, prev_V - V
end

function step!(
    ordering,
    p,
    prob::Vector{<:StateIntervalProbabilities},
    prev_V,
    V,
    indices;
    max,
)
    partial_ominmax!(ordering, p, prob, V, indices; max = max)

    @inbounds for j in indices
        V[j] = dot(p[j], prev_V)
    end
end

function step!(ordering, p, prob::MatrixIntervalProbabilities, prev_V, V, indices; max)
    partial_ominmax!(ordering, p, prob, V, indices; max = max)

    res = transpose(transpose(prev_V) * p)
    V[indices] .= res[indices]
end
