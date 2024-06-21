struct InvalidStateError{T1 <: UnionIndex, T2 <: Union{<:UnionIndex, <:AbstractVector}} <:
       Exception
    invalid_state::T1
    valid_states::T2
end

function Base.showerror(
    io::IO,
    e::InvalidStateError{T, Union{<:Tuple, <:AbstractVector}},
) where {T}
    print(io, "state $(e.invalid_state) is invalid. Valid states are (")
    for (i, ax) in enumerate(e.valid_states)
        print(io, "1:", ax)
        if i < length(e.valid_states)
            print(io, ", ")
        end
    end
    print(io, ").")
end

Base.showerror(io::IO, e::InvalidStateError{T, <:Integer}) where {T} =
    print(io, "state $(e.invalid_state) is invalid. Valid states are 1:$(e.valid_states).")

struct StateDimensionMismatch <: Exception
    invalid_state::Tuple
    state_dim::Integer
end

Base.showerror(io::IO, e::StateDimensionMismatch) = print(
    io,
    "state dimension $(length(e.invalid_state)) of the state $(e.invalid_state) does not match the system dimension $(e.state_dim).",
)
