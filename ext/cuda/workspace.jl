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


######################
# Factored workspace #
######################
struct CuFactoredOMaxWorkspace{N, M} <: AbstractCuWorkspace
    max_support_per_marginal::NTuple{N, Int32}
    workspace_partitioning::NTuple{M, Int32}
end

function CuFactoredOMaxWorkspace(sys::IntervalMDP.FactoredRMDP)
    max_support_per_marginal =
        Tuple(Int32(IntervalMDP.maxsupportsize(ambiguity_sets(marginal))) for marginal in marginals(sys))
    workspace_partitioning = (one(Int32), Int32.(cumsum(max_support_per_marginal) .+ 1)...)
    return CuFactoredOMaxWorkspace(max_support_per_marginal, workspace_partitioning)
end

IntervalMDP.construct_workspace(
    sys::IntervalMDP.FactoredRMDP,
    marginal::Marginal{<:IntervalAmbiguitySets{R, MR}},
    ::IntervalMDP.IsFIMDP,
    ::OMaximization;
    kwargs...,
) where {R, MR <:AbstractGPUMatrix{R}} = CuFactoredOMaxWorkspace(sys)

IntervalMDP.construct_workspace(
    sys::IntervalMDP.FactoredRMDP,
    marginal::Marginal{<:IntervalAmbiguitySets{R, MR}},
    ::IntervalMDP.IsFIMDP,
    ::OMaximization;
    kwargs...,
) where {R, MR <:AbstractCuSparseMatrix{R}} = CuFactoredOMaxWorkspace(sys)
