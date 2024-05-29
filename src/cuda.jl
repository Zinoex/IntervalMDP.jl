# NOTE: This file does not explicitly use the CUDA.jl package, but it is
# included to allow the CUDA extension to define the necessary functions and types.

struct CuModelAdaptor{Tv} end

valtype(::Type{CuModelAdaptor{Tv}}) where {Tv} = Tv

function cu end
