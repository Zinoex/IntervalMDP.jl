struct StateIntervalProbabilities{R, VR <: AbstractVector{R}}
    lower::VR
    gap::VR

    sum_lower::R
end

function StateIntervalProbabilities(lower::VR, gap::VR) where {R, VR <: AbstractVector{R}}
    joint_lower_bound = sum(lower)
    @assert joint_lower_bound <= 1 "The joint lower bound transition probability (is $joint_lower_bound) should be less than or equal to 1."

    joint_upper_bound = joint_lower_bound + sum(gap)
    @assert joint_upper_bound >= 1 "The joint upper bound transition probability (is $joint_upper_bound) should be greater than or equal to 1."

    return StateIntervalProbabilities(lower, gap, joint_lower_bound)
end

# Keyword constructor
function StateIntervalProbabilities(; lower::VR, upper::VR) where {VR <: AbstractVector}
    lower, gap = compute_gap(lower, upper)
    return StateIntervalProbabilities(lower, gap)
end

function compute_gap(lower::VR, upper::VR) where {VR <: AbstractVector}
    gap = upper - lower
    return lower, gap
end

gap(s::StateIntervalProbabilities) = s.gap
lower(s::StateIntervalProbabilities) = s.lower
sum_lower(s::StateIntervalProbabilities) = s.sum_lower

gap(V::Vector{<:StateIntervalProbabilities}) = gap.(V)
num_src(V::Vector{<:StateIntervalProbabilities}) = length(V)

interval_prob_hcat(transition_probs::Vector{<:Vector{<:StateIntervalProbabilities}}) = reduce(vcat, transition_probs)


struct MatrixIntervalProbabilities{R, VR <: AbstractVector{R}, MR <: AbstractMatrix{R}}
    lower::MR
    gap::MR

    sum_lower::VR
end

function MatrixIntervalProbabilities(lower::MR, gap::MR) where {R, MR <: AbstractMatrix{R}}
    sum_lower = vec(sum(lower; dims = 1))

    max_lower_bound = maximum(sum_lower)
    @assert max_lower_bound <= 1 "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."

    sum_upper = vec(sum(lower + gap; dims = 1))

    max_upper_bound = minimum(sum_upper)
    @assert max_upper_bound >= 1 "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."

    return MatrixIntervalProbabilities(lower, gap, sum_lower)
end

# Keyword constructor
function MatrixIntervalProbabilities(; lower::MR, upper::MR) where {MR <: AbstractMatrix}
    lower, gap = compute_gap(lower, upper)
    return MatrixIntervalProbabilities(lower, gap)
end

function compute_gap(lower::MR, upper::MR) where {MR <: AbstractMatrix}
    gap = upper - lower
    return lower, gap
end

gap(s::MatrixIntervalProbabilities) = s.gap
lower(s::MatrixIntervalProbabilities) = s.lower
sum_lower(s::MatrixIntervalProbabilities) = s.sum_lower
num_src(s::MatrixIntervalProbabilities) = size(gap(s), 2)

function interval_prob_hcat(transition_probs::Vector{<:MatrixIntervalProbabilities})
    l = mapreduce(lower, hcat, transition_probs)
    g = mapreduce(gap, hcat, transition_probs)

    sl = mapreduce(sum_lower, vcat, transition_probs)

    return MatrixIntervalProbabilities(l, g, sl)
end

const IntervalProbabilities{R} =
    Union{Vector{<:StateIntervalProbabilities{R}}, <:MatrixIntervalProbabilities{R}}
