@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} =
    maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))

arrayfactory(mp::ProductProcess, T, sizes) =
    arrayfactory(markov_process(mp), T, sizes)
arrayfactory(mp::FactoredRMDP, T, sizes) =
    arrayfactory(marginals(mp)[1], T, sizes)
arrayfactory(marginal::Marginal, T, sizes) =
    arrayfactory(ambiguity_sets(marginal), T, sizes)
arrayfactory(prob::IntervalAmbiguitySets, T, sizes) =
    arrayfactory(prob.gap, T, sizes)
arrayfactory(::MR, T, sizes) where {MR <: AbstractArray} = Array{T}(undef, sizes)

function valuetype(prob::AbstractIntervalMDPProblem)
    spec_valuetype = valuetype(specification(prob))
    sys_valuetype = valuetype(system(prob))

    if isnothing(spec_valuetype)
        return sys_valuetype
    end

    return promote_type(spec_valuetype, sys_valuetype)
end
valuetype(spec::Specification) = valuetype(system_property(spec))

valuetype(mp::ProductProcess) = valuetype(markov_process(mp))
valuetype(mp::FactoredRMDP) = promote_type(valuetype.(marginals(mp))...)
valuetype(marginal::Marginal) = valuetype(ambiguity_sets(marginal))
valuetype(::IntervalAmbiguitySets{R}) where {R} = R
valuetype(::IntervalAmbiguitySet{R}) where {R} = R
valuetype(::AbstractArray{R}) where {R} = R

valuetype(::TimeVaryingStrategy{N, <:AbstractArray{NTuple{N, T}}}) where {N, T} = T
valuetype(::StationaryStrategy{N, <:AbstractArray{NTuple{N, T}}}) where {N, T} = T
valuetype(::NoStrategy) = nothing

valuetype(::Property) = nothing
valuetype(::FiniteTimeReward{R}) where {R} = R
valuetype(::InfiniteTimeReward{R}) where {R} = R