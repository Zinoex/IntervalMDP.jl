abstract type AbstractCuWorkspace end

###################
# Dense workspace #
###################
struct CuDenseOMaxWorkspace <: AbstractCuWorkspace
    num_actions::Int32
end

IntervalMDP.construct_workspace(
    prob::IntervalAmbiguitySets{R, MR},
    ::OMaximization = IntervalMDP.default_bellman_algorithm(prob);
    num_actions = 1,
    kwargs...,
) where {R, MR <: AbstractGPUMatrix{R}} = CuDenseOMaxWorkspace(num_actions)

####################
# Sparse workspace #
####################
struct CuSparseOMaxWorkspace <: AbstractCuWorkspace
    max_support::Int32
    num_actions::Int32
end

function CuSparseOMaxWorkspace(p::IntervalAmbiguitySets, num_actions)
    max_support = IntervalMDP.maxsupportsize(p)
    return CuSparseOMaxWorkspace(max_support, num_actions)
end

IntervalMDP.construct_workspace(
    prob::IntervalAmbiguitySets{R, MR},
    ::OMaximization = IntervalMDP.default_bellman_algorithm(prob);
    num_actions = 1,
    kwargs...,
) where {R, MR <: AbstractCuSparseMatrix{R}} = CuSparseOMaxWorkspace(prob, num_actions)
