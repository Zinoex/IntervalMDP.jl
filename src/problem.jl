abstract type AbstractIntervalMDPProblem end

################
# Verification #
#################

"""
    VerificationProblem{S <: StochasticProcess, F <: Specification, C <: AbstractStrategy}

A verification problem is a tuple of an interval Markov process and a specification.

### Fields
- `system::S`: interval Markov process.
- `spec::F`: specification (either temporal logic or reachability-like).
- `strategy::C`: strategy to be used for verification, which can be a given strategy or a no strategy, 
  i.e. select (but do not store! see [`ControlSynthesisProblem`]) optimal action for every state at every timestep.
"""
struct VerificationProblem{
    S <: StochasticProcess,
    F <: Specification,
    C <: AbstractStrategy,
} <: AbstractIntervalMDPProblem
    system::S
    spec::F
    strategy::C

    function VerificationProblem(
        system::S,
        spec::F,
        strategy::C,
    ) where {S <: StochasticProcess, F <: Specification, C <: AbstractStrategy}
        checkspecification(spec, system, strategy)
        checkstrategy(strategy, system)
        return new{S, F, C}(system, spec, strategy)
    end
end

VerificationProblem(system::StochasticProcess, spec::Specification) =
    VerificationProblem(system, spec, NoStrategy())

"""
    system(prob::VerificationProblem)

Return the system of a problem.
"""
system(prob::VerificationProblem) = prob.system

"""
    specification(prob::VerificationProblem)

Return the specification of a problem.
"""
specification(prob::VerificationProblem) = prob.spec

"""
    strategy(prob::VerificationProblem)

Return the strategy of a problem, if provided.
"""
strategy(prob::VerificationProblem) = prob.strategy

struct VerificationSolution{R, MR <: AbstractArray{R}, D}
    value_function::MR
    residual::MR
    num_iterations::Int
    additional_data::D
end
VerificationSolution(V, res, k) = VerificationSolution(V, res, k, nothing)

"""
    value_function(s::VerificationSolution)

Return the value function of a verification solution.
"""
value_function(s::VerificationSolution) = s.value_function

"""
    residual(s::VerificationSolution)

Return the residual of a verification solution.
"""
residual(s::VerificationSolution) = s.residual

"""
    num_iterations(s::VerificationSolution)

Return the number of iterations of a verification solution.
"""
num_iterations(s::VerificationSolution) = s.num_iterations

Base.iterate(s::VerificationSolution, args...) =
    iterate((s.value_function, s.num_iterations, s.residual), args...)

#####################
# Control synthesis #
#####################

"""
    ControlSynthesisProblem{S <: StochasticProcess, F <: Specification}

A verification problem is a tuple of an interval Markov process and a specification.

### Fields
- `system::S`: interval Markov process.
- `spec::F`: specification (either temporal logic or reachability-like).
"""
struct ControlSynthesisProblem{S <: StochasticProcess, F <: Specification} <:
       AbstractIntervalMDPProblem
    system::S
    spec::F

    function ControlSynthesisProblem(
        system::S,
        spec::F,
    ) where {S <: StochasticProcess, F <: Specification}
        checkspecification(spec, system)
        return new{S, F}(system, spec)
    end
end

"""
    system(prob::ControlSynthesisProblem)

Return the system of a problem.
"""
system(prob::ControlSynthesisProblem) = prob.system

"""
    specification(prob::ControlSynthesisProblem)

Return the specification of a problem.
"""
specification(prob::ControlSynthesisProblem) = prob.spec

struct ControlSynthesisSolution{C <: AbstractStrategy, R, MR <: AbstractArray{R}, D}
    strategy::C
    value_function::MR
    residual::MR
    num_iterations::Int
    additional_data::D
end
ControlSynthesisSolution(strategy, V, res, k) =
    ControlSynthesisSolution(strategy, V, res, k, nothing)

"""
    strategy(s::ControlSynthesisSolution)

Return the strategy of a control synthesis solution.
"""
strategy(s::ControlSynthesisSolution) = s.strategy

"""
    value_function(s::ControlSynthesisSolution)

Return the value function of a control synthesis solution.
"""
value_function(s::ControlSynthesisSolution) = s.value_function

"""
    residual(s::ControlSynthesisSolution)

Return the residual of a control synthesis solution.
"""
residual(s::ControlSynthesisSolution) = s.residual

"""
    num_iterations(s::ControlSynthesisSolution)

Return the number of iterations of a control synthesis solution.
"""
num_iterations(s::ControlSynthesisSolution) = s.num_iterations

Base.iterate(s::ControlSynthesisSolution, args...) =
    iterate((s.strategy, s.value_function, s.num_iterations, s.residual), args...)
