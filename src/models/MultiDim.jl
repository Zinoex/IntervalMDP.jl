# TODO: Document
struct MultiDim{N, P <: SimpleIntervalMarkovProcess} <: SequentialIntervalMarkovProcess
    underlying_process::P
    num_states::Int32
    dims::NTuple{N, <:Integer}

    function MultiDim(underlying_process::P, dims::NTuple{N, <:Integer}) where {N, P <: SimpleIntervalMarkovProcess}
        nstates = prod(dims)
        if num_states(underlying_process) != nstates
            throw(ArgumentError("Number of states $(num_states(underlying_process)) in the underlying process must be equal to the product of the dimensions $dims"))
        end

        new{N, P}(underlying_process, nstates, dims)
    end
end

dims(::MultiDim{N}) where {N} = N
product_num_states(mp::MultiDim) = collect(mp.dims)
transition_matrix_type(mp::MultiDim) = transition_matrix_type(mp.underlying_process)