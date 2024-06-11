struct InvalidStateError{T <: UnionIndex} <: Exception 
    invalid_state::T
    valid_states::T
end

function Base.showerror(io::IO, e::InvalidStateError{<:Tuple})
    print(io, "state $(e.invalid_state) is invalid. Valid states are (")
    for (i, ax) in enumerate(e.valid_states)
        print(io, "1:", ax)
        if i < length(e.valid_states)
            print(io, ", ")
        end
    end
    print(io, ").")
end

Base.showerror(io::IO, e::InvalidStateError{<:Integer}) = print(io, "state $(e.invalid_state) is invalid. Valid states are 1:$(e.valid_states).")

struct StateDimensionMismatch <: Exception 
    invalid_state::Tuple
    valid_states::Tuple
end

Base.showerror(io::IO, e::StateDimensionMismatch) = 
    print(io, "state dimension $(length(e.invalid_state)) of $(e.invalid_state) does not match the system dimension $(length(e.valid_states)).")
