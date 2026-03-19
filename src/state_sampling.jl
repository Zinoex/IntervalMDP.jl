
exhaustive_cartesian(model::FactoredRMDP) = exhaustive_cartesian(model, modeltype(model))
exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP) = CartesianIndices(source_shape(model))
exhaustive_cartesian(model::IntervalAmbiguitySets) = CartesianIndices(source_shape(model))


function custom_sequence(model::FactoredRMDP, sequence::Vector{NTuple{M, T}})::Array{CartesianIndex, 1} where {M, T}
    # verify sequence valid for model
    @assert all(x -> all(>=(1), x), sequence)
    @assert all(x -> all(x .<= source_shape(model)), sequence)
    
    # convert to iterator of CartesianIndex
    return [CartesianIndex(s) for s in sequence]
end

# TODO: 1. random sampling of states, with or without replacement, with or without weighting (e.g. based on current value function)
# TODO:     - subset of states each iteration?
# TODO:     - one state per iteration?
# TODO:
# TODO: 2. (epsilon) greedy on policy trajectory simulation
# TODO: 3. BRTDP gap based trajectory simulation
# TODO: 
