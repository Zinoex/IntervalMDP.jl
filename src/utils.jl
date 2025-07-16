@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} =
    maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
@inline @inbounds mindiff(x::V) where {V <: AbstractVector} =
    minimum(x[i + 1] - x[i] for i in 1:(length(x) - 1))

arrayfactory(mp::ProductProcess, T, num_states) =
    arrayfactory(markov_process(mp), T, num_states)
arrayfactory(mp::IntervalMarkovProcess, T, num_states) =
    arrayfactory(transition_prob(mp), T, num_states)
arrayfactory(prob::MixtureIntervalProbabilities, T, num_states) =
    arrayfactory(first(prob), T, num_states)
arrayfactory(prob::OrthogonalIntervalProbabilities, T, num_states) =
    arrayfactory(first(prob), T, num_states)
arrayfactory(prob::IntervalProbabilities, T, num_states) =
    arrayfactory(gap(prob), T, num_states)
arrayfactory(::MR, T, num_states) where {MR <: AbstractArray} = zeros(T, num_states)

valuetype(mp::ProductProcess) = valuetype(markov_process(mp))
valuetype(mp::IntervalMarkovProcess) = valuetype(transition_prob(mp))
valuetype(prob::MixtureIntervalProbabilities) = valuetype(first(prob))
valuetype(prob::OrthogonalIntervalProbabilities) = valuetype(first(prob))
valuetype(prob::IntervalProbabilities) = valuetype(gap(prob))
valuetype(::MR) where {R, MR <: AbstractArray{R}} = R
