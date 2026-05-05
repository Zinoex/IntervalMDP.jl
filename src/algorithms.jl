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

# `showmcalgorithm` for the model-pretty-print path. Concrete algorithms
# (`RobustValueIteration`, `GeneralizedSamplingbasedRobustDynamicProgramming`)
# overload these in their own files; the fallback below covers
# any future algorithm that hasn't yet supplied a show.
function showmcalgorithm(io::IO, prefix, ::ModelCheckingAlgorithm)
    println(io, prefix, "├─", styled"Default model checking algorithm: {green:None}")
end

# `solve(problem)` (no algorithm) defaults to `RobustValueIteration` with
# the model's default Bellman algorithm. Defined here as a forward
# declaration so this file can be `include`d before `RobustValueIteration`
# is defined; the actual `default_algorithm` body refers to types that
# come later, but it's only resolved at call time.
default_algorithm(problem::AbstractIntervalMDPProblem) = default_algorithm(system(problem))
default_algorithm(system::StochasticProcess) =
    RobustValueIteration(default_bellman_algorithm(system))

solve(problem::AbstractIntervalMDPProblem; kwargs...) =
    solve(problem, default_algorithm(problem); kwargs...)
