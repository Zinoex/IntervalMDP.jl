# TODO: Document
struct MultiDim{N, P <: SimpleIntervalMarkovProcess} <: SequentialIntervalMarkovProcess
    underlying_process::P
    dims::NTuple{N, Int32}

    function MultiDim(underlying_process::P, dims::NTuple{N, Int32}) where {N, P <: SimpleIntervalMarkovProcess}
        if !num_states(underlying_process) == prod(dims)
            throw(ArgumentError("Number of states $(num_states(underlying_process)) in the underlying process must be equal to the product of the dimensions $dims"))
        end

        new{N, P}(underlying_process, dims)
    end
end

dims(::MultiDim{N}) where {N} = N
product_num_states(mp::MultiDim) = Vector(mp.dims)
first_transition_prob(mp::MultiDim) = first_transition_prob(mp.underlying_process)