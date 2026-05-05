abstract type BellmanAlgorithm end
struct OMaximization <: BellmanAlgorithm end
Base.@kwdef struct LPMcCormickRelaxation{O} <: BellmanAlgorithm
    lp_solver::O = HiGHS.Optimizer
end
struct VertexEnumeration <: BellmanAlgorithm end
struct UnknownBellmanAlgorithm <: BellmanAlgorithm end

default_bellman_algorithm(pp::ProductProcess) =
    default_bellman_algorithm(markov_process(pp))
default_bellman_algorithm(mdp::FactoredRMDP) =
    default_bellman_algorithm(mdp, modeltype(mdp))
default_bellman_algorithm(::FactoredRMDP, ::IsIMDP) = OMaximization()
default_bellman_algorithm(::FactoredRMDP, ::IsFIMDP) = OMaximization()
default_bellman_algorithm(::IntervalAmbiguitySets) = OMaximization()
default_bellman_algorithm(_, ::Union{IsRMDP, IsFRMDP}) = UnknownBellmanAlgorithm()

function showbellmanalg(io::IO, prefix, ::IsIMDP, ::OMaximization)
    println(
        io,
        prefix,
        "└─",
        styled"Default Bellman operator algorithm: {green:O-Maximization}",
    )
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP, ::OMaximization)
    println(
        io,
        prefix,
        "└─",
        styled"Default Bellman operator algorithm: {green:Recursive O-Maximization}",
    )
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP, ::LPMcCormickRelaxation)
    println(
        io,
        prefix,
        "└─",
        styled"Default Bellman operator algorithm: {green:Binary tree LP McCormick Relaxation}",
    )
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP, ::VertexEnumeration)
    println(
        io,
        prefix,
        "└─",
        styled"Default Bellman operator algorithm: {green:Vertex Enumeration}",
    )
end

function showbellmanalg(io::IO, prefix, _, ::BellmanAlgorithm)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:None}")
end

abstract type ModelCheckingAlgorithm end
