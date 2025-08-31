arrayfactory(mp::ProductProcess, T, num_states) =
    arrayfactory(markov_process(mp), T, num_states)
arrayfactory(mp::FactoredRMDP, T, num_states) =
    arrayfactory(mp.transition[1], T, num_states)
arrayfactory(marginal::Marginal, T, num_states) =
    arrayfactory(marginal.ambiguity_sets, T, num_states)
arrayfactory(prob::IntervalAmbiguitySets, T, num_states) =
    arrayfactory(prob.gap, T, num_states)
arrayfactory(::MR, T, num_states) where {MR <: AbstractArray} = zeros(T, num_states)

valuetype(mp::ProductProcess) = valuetype(markov_process(mp))
valuetype(mp::FactoredRMDP) = valuetype(mp.transition[1])
valuetype(marginal::Marginal) = valuetype(marginal.ambiguity_sets)
valuetype(prob::IntervalAmbiguitySets) = valuetype(prob.gap)
valuetype(::MR) where {R, MR <: AbstractArray{R}} = R
