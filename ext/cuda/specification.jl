
Adapt.@adapt_structure FiniteTimeReward
Adapt.@adapt_structure InfiniteTimeReward
Adapt.@adapt_structure Specification
Adapt.@adapt_structure Problem

function IntervalMDP.checkdevice!(::AbstractGPUArray, ::AbstractGPUMatrix)
    # Both arguments are on the GPU.
    return nothing
end

function IntervalMDP.checkdevice!(::AbstractGPUArray, ::AbstractCuSparseMatrix)
    # Both arguments are on the GPU.
    return nothing
end

function IntervalMDP.checkdevice!(b::AbstractArray, A::AbstractGPUMatrix)
    # The first argument is on the CPU (technically in RAM) and the second is on the GPU.
    @assert false "The reward vector is a CPU array ($(typeof(b))) and the transition matrix is on the GPU ($(typeof(A)))."
end

function IntervalMDP.checkdevice!(b::AbstractArray, A::AbstractCuSparseMatrix)
    # The first argument is on the CPU (technically in RAM) and the second is on the GPU.
    @assert false "The reward vector is a CPU array ($(typeof(b))) and the transition matrix is on the GPU ($(typeof(A)))."
end

function IntervalMDP.checkdevice!(b::AbstractGPUArray, A::AbstractMatrix)
    # The first argument is on the GPU and the second is on the CPU (technically in RAM).
    @assert false "The reward vector is on the GPU ($(typeof(b))) and the transition matrix is a CPU matrix ($(typeof(A)))."
end
