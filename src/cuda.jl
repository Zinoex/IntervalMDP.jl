# NOTE: This file does not explicitly use the CUDA.jl package, but it is
# included to allow the CUDA extension to define the necessary functions and types.

struct CuModelAdaptor{Tv} end
struct CpuModelAdaptor{Tv} end

valtype(::Type{CuModelAdaptor{Tv}}) where {Tv} = Tv
valtype(::Type{CpuModelAdaptor{Tv}}) where {Tv} = Tv

function cu end
function cpu end

struct OutOfSharedMemory <: Exception
    min_shared_memory::Int
end

function Base.showerror(io::IO, e::OutOfSharedMemory)
    println(
        io,
        "Out of shared memory: minimum required shared memory for the problem is ",
        e.min_shared_memory,
        " bytes.",
    )
    println(
        io,
        "Please try either the CPU implementation, the (dense) decomposed representation (preferred), or use a larger GPU.",
    )
end

function checkdevice(v::AbstractArray, system::FactoredRMDP)
    for marginal in system.transition
        checkdevice(v, marginal)
    end
end

function checkdevice(v::AbstractArray, marginal::Marginal)
    checkdevice(v, ambiguity_sets(marginal))
end

function checkdevice(v::AbstractArray, p::IntervalAmbiguitySets)
    # Lower and gap are required to be the same type, so not necessary to check.
    checkdevice(v, p.lower)
end

function checkdevice(::AbstractArray, ::AbstractMatrix)
    # Both arguments are on the CPU (technically in RAM).
    return nothing
end
