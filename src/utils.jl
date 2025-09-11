arrayfactory(mp::ProductProcess, T, sizes) =
    arrayfactory(markov_process(mp), T, sizes)
arrayfactory(mp::FactoredRMDP, T, sizes) =
    arrayfactory(marginals(mp)[1], T, sizes)
arrayfactory(marginal::Marginal, T, sizes) =
    arrayfactory(ambiguity_sets(marginal), T, sizes)
arrayfactory(prob::IntervalAmbiguitySets, T, sizes) =
    arrayfactory(prob.gap, T, sizes)
arrayfactory(::MR, T, sizes) where {MR <: AbstractArray} = Array{T}(undef, sizes)

valuetype(mp::ProductProcess) = valuetype(markov_process(mp))
valuetype(mp::FactoredRMDP) = promote_type(valuetype.(marginals(mp))...)
valuetype(marginal::Marginal) = valuetype(ambiguity_sets(marginal))
valuetype(::IntervalAmbiguitySets{R}) where {R} = R
valuetype(::AbstractArray{R}) where {R} = R
valuetype(::FiniteTimeReward{R}) where {R} = R
valuetype(::InfiniteTimeReward{R}) where {R} = R