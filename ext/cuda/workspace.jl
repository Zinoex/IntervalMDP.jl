abstract type AbstractCuWorkspace end

###################
# Dense workspace #
###################
struct CuDenseWorkspace <: AbstractCuWorkspace
    max_actions::Int32
end

IntervalMDP.construct_workspace(::AbstractGPUMatrix, max_actions) =
    CuDenseWorkspace(max_actions)

# Product workspace
struct CuDenseProductWorkspace <: IntervalMDP.CompositeWorkspace
    max_actions::Int32
    state_index::Int32
end

function IntervalMDP._construct_workspace(
    ::AbstractGPUMatrix,
    mp::SimpleIntervalMarkovProcess,
    state_index,
)
    mactions = IntervalMDP.max_actions(mp)

    return CuDenseProductWorkspace(mactions, state_index), state_index + one(Int32)
end

####################
# Sparse workspace #
####################
struct CuSparseWorkspace <: AbstractCuWorkspace
    max_nonzeros::Int32
    max_actions::Int32
end

function CuSparseWorkspace(p::AbstractCuSparseMatrix, max_actions)
    max_nonzeros = maximum(nnz, eachcol(p))
    return CuSparseWorkspace(max_nonzeros, max_actions)
end

IntervalMDP.construct_workspace(p::AbstractCuSparseMatrix, max_actions) =
    CuSparseWorkspace(p, max_actions)

# Product workspace
struct CuSparseProductWorkspace <: IntervalMDP.CompositeWorkspace
    max_actions::Int32
    state_index::Int32
end

function CuSparseProductWorkspace(p::AbstractCuSparseMatrix, max_actions, state_index)
    max_nonzeros = maximum(nnz, eachcol(p))
    return CuSparseProductWorkspace(max_nonzeros, max_actions, state_index)
end

function IntervalMDP._construct_workspace(
    p::AbstractCuSparseMatrix,
    mp::SimpleIntervalMarkovProcess,
    state_index,
)
    mactions = IntervalMDP.max_actions(mp)

    return CuSparseProductWorkspace(p, mactions, state_index), state_index + one(Int32)
end
