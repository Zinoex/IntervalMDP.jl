
Adapt.@adapt_structure FiniteTimeReward
Adapt.@adapt_structure InfiniteTimeReward
Adapt.@adapt_structure Specification
Adapt.@adapt_structure Problem

function IntervalMDP.checkdevice!(::AbstractGPUVector, ::AbstractGPUMatrix)
    # Both arguments are on the GPU.
    return nothing
end

function IntervalMDP.checkdevice!(::AbstractVector, ::AbstractGPUMatrix)
    # The first argument is on the CPU (technically in RAM) and the second is on the GPU.
    @assert false "The reward vector is a CPU array and the transition matrix is on the GPU."
end

function IntervalMDP.checkdevice!(::AbstractGPUVector, ::AbstractMatrix)
    # The first argument is on the GPU and the second is on the CPU (technically in RAM).
    @assert false "The reward vector is on the GPU and the transition matrix is a CPU matrix."
end
