arrayfactory(mp::ProductProcess, T, num_states) =
    arrayfactory(markov_process(mp), T, num_states)
arrayfactory(mp::FactoredRMDP, T, num_states) =
    arrayfactory(marginals(mp)[1], T, num_states)
arrayfactory(marginal::Marginal, T, num_states) =
    arrayfactory(ambiguity_sets(marginal), T, num_states)
arrayfactory(prob::IntervalAmbiguitySets, T, num_states) =
    arrayfactory(prob.gap, T, num_states)
arrayfactory(::MR, T, num_states) where {MR <: AbstractArray} = Array{T}(undef, num_states)

valuetype(mp::ProductProcess) = valuetype(markov_process(mp))
valuetype(mp::FactoredRMDP) = promote_type(valuetype.(marginals(mp))...)
valuetype(marginal::Marginal) = valuetype(ambiguity_sets(marginal))
valuetype(::IntervalAmbiguitySets{R}) where {R} = R
