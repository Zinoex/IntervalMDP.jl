# NOTE: This file does not explicitly use the CUDA.jl package, but it is
# included to allow the CUDA extension to define the necessary functions and types.

struct CuModelAdaptor{Tv, Ti} end

valtype(::Type{CuModelAdaptor{Tv, Ti}}) where {Tv, Ti} = Tv
indtype(::Type{CuModelAdaptor{Tv, Ti}}) where {Tv, Ti} = Ti

function cu end
