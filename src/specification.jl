### Specification types

"""
    Specification

Super type for all system specficiations
"""
abstract type Specification end

## Temporal logics

"""
    AbstractTemporalLogic

Super type for temporal logic specifications
"""
abstract type AbstractTemporalLogic <: Specification end

"""
    LTLFormula

Linear Temporal Logic (LTL) specifications (first-order logic + next and until operators) [1].
Let ``ϕ`` denote the formula and ``M`` denote an interval Markov process. Then compute ``M ⊧ ϕ``.

[1] Vardi, M.Y. (1996). An automata-theoretic approach to linear temporal logic. In: Moller, F., Birtwistle, G. (eds) Logics for Concurrency. Lecture Notes in Computer Science, vol 1043. Springer, Berlin, Heidelberg.
"""
struct LTLFormula <: AbstractTemporalLogic
    formula::String
end

"""
    isfinitetime(spec::LTLFormula)

Return `false` for an LTL formula. LTL formulas are not finite time specifications.
"""
isfinitetime(spec::LTLFormula) = false


"""
    LTLfFormula

An LTL formula over finite traces [1]. See [`LTLFormula`](@ref) for the structure of LTL formulas.
Let ``ϕ`` denote the formula, ``M`` denote an interval Markov process, and ``H`` the time horizon.
Then compute ``M ⊧ ϕ`` within traces of length ``H``.

### Fields
- `formula::String`: LTL formula
- `time_horizon::T`: Time horizon of the finite traces 

[1] Giuseppe De Giacomo and Moshe Y. Vardi. 2013. Linear temporal logic and linear dynamic logic on finite traces. In Proceedings of the Twenty-Third international joint conference on Artificial Intelligence (IJCAI '13). AAAI Press, 854–860.
"""
struct LTLfFormula{T <: Integer} <: AbstractTemporalLogic
    formula::String
    time_horizon::T
end

"""
    isfinitetime(spec::LTLfFormula)

Return `true` for an LTLf formula. LTLf formulas are specifically over finite traces.
"""
isfinitetime(spec::LTLfFormula) = true

"""
    time_horizon(spec::LTLfFormula)

Return the time horizon of an LTLf formula.
"""
time_horizon(spec::LTLfFormula) = spec.time_horizon

"""
    PCTLFormula

A Probabilistic Computation Tree Logic (PCTL) formula [1].
Let ``ϕ`` denote the formula and ``M`` denote an interval Markov process. Then compute ``M ⊧ ϕ``.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
struct PCTLFormula <: AbstractTemporalLogic
    formula::String
end

## Reachability

"""
    AbstractReachability

Super type for all reachability-like specifications.
"""
abstract type AbstractReachability <: Specification end

"""
    FiniteTimeReachability{T <: Integer, VT <: AbstractVector{T}}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, if ``T`` is the set of target states and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T)``.
"""
struct FiniteTimeReachability{T <: Integer, VT <: AbstractVector{T}} <: AbstractReachability
    terminal_states::VT
    time_horizon::Any
end

function checkspecification!(spec::FiniteTimeReachability, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(spec), num_states(system))
end

"""
    isfinitetime(spec::FiniteTimeReachability)

Return `true` for FiniteTimeReachability.
"""
isfinitetime(spec::FiniteTimeReachability) = true

"""
    time_horizon(spec::FiniteTimeReachability)

Return the time horizon of a finite time reachability specification.
"""
time_horizon(spec::FiniteTimeReachability) = spec.time_horizon

"""
    terminal_states(spec::FiniteTimeReachability)

Return the set of terminal states of a finite time reachability specification.
"""
terminal_states(spec::FiniteTimeReachability) = spec.terminal_states

"""
    terminal_states(spec::FiniteTimeReachability)

Return the set of states with which to compute reachbility for a finite time reachability specification.
This is equivalent for [`terminal_states(spec::FiniteTimeReachability)`](@ref) for a regular reachability
property. See [`FiniteTimeReachAvoid`](@ref) for a more complex specification where the reachability and
terminal states differ.
"""
reach(spec::FiniteTimeReachability) = spec.terminal_states

"""
    InfiniteTimeReachability{R <: Real, T <: Integer, VT <: AbstractVector{T}} 
 
`InfiniteTimeReachability` is similar to [`FiniteTimeReachability`](@ref) except that the time horizon is infinite.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `eps`.
"""
struct InfiniteTimeReachability{R <: Real, T <: Integer, VT <: AbstractVector{T}} <: AbstractReachability
    terminal_states::VT
    eps::R
end

function checkspecification!(spec::InfiniteTimeReachability, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(spec), num_states(system))
end

"""
    isfinitetime(spec::InfiniteTimeReachability)

Return `false` for InfiniteTimeReachability.
"""
isfinitetime(spec::InfiniteTimeReachability) = false

"""
    eps(spec::InfiniteTimeReachability)

Return the convergence threshold of an infinite time reachability specification.
"""
eps(spec::InfiniteTimeReachability) = spec.eps

"""
    terminal_states(spec::InfiniteTimeReachability)

Return the set of terminal states of an infinite time reachability specification.
"""
terminal_states(spec::InfiniteTimeReachability) = spec.terminal_states

"""
    reach(spec::InfiniteTimeReachability)

Return the set of states with which to compute reachbility for a infinite time reachability specification.
This is equivalent for [`terminal_states(spec::InfiniteTimeReachability)`](@ref) for a regular reachability
property. See [`InfiniteTimeReachAvoid`](@ref) for a more complex specification where the reachability and
terminal states differ.
"""
reach(spec::InfiniteTimeReachability) = spec.terminal_states

## Reach-avoid

"""
    AbstractReachAvoid

A specialization of reachability that includes a set of states to avoid.
"""
abstract type AbstractReachAvoid <: AbstractReachability end

"""
    FiniteTimeReachAvoid{T <: Integer, VT <: AbstractVector{T}}

Finite time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, if ``T`` is the set of target states, ``A`` is the set of states to avoid, and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T and ∀k' = 0…k, s_k' ∉ A)``.
"""
struct FiniteTimeReachAvoid{T <: Integer, VT <: AbstractVector{T}} <: AbstractReachAvoid
    reach::VT
    avoid::VT
    time_horizon::Any
end

function checkspecification!(spec::FiniteTimeReachAvoid, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(spec), num_states(system))
    checkdisjoint!(reach(spec), avoid(spec))
end

"""
    isfinitetime(spec::FiniteTimeReachAvoid)

Return `true` for FiniteTimeReachAvoid.
"""
isfinitetime(spec::FiniteTimeReachAvoid) = true

"""
    time_horizon(spec::FiniteTimeReachAvoid)

Return the time horizon of a finite time reach-avoid specification.
"""
time_horizon(spec::FiniteTimeReachAvoid) = spec.time_horizon

"""
    terminal_states(spec::FiniteTimeReachAvoid)

Return the set of terminal states of a finite time reach-avoid specification.
That is, the union of the reach and avoid sets.
"""
terminal_states(spec::FiniteTimeReachAvoid) = [spec.reach; spec.avoid]

"""
    reach(spec::FiniteTimeReachAvoid)

Return the set of target states.
"""
reach(spec::FiniteTimeReachAvoid) = spec.reach

"""
    avoid(spec::FiniteTimeReachAvoid)

Return the set of states to avoid.
"""
avoid(spec::FiniteTimeReachAvoid) = spec.avoid

"""
    InfiniteTimeReachAvoid{R <: Real, T <: Integer, VT <: AbstractVector{T}}

`InfiniteTimeReachAvoid` is similar to [`FiniteTimeReachAvoid`](@ref) except that the time horizon is infinite.
"""
struct InfiniteTimeReachAvoid{R <: Real, T <: Integer, VT <: AbstractVector{T}} <: AbstractReachAvoid
    reach::VT
    avoid::VT
    eps::R
end

function checkspecification!(spec::InfiniteTimeReachAvoid, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(spec), num_states(system))
    checkdisjoint!(reach(spec), avoid(spec))
end

"""
    isfinitetime(spec::InfiniteTimeReachAvoid)

Return `false` for InfiniteTimeReachAvoid.
"""
isfinitetime(spec::InfiniteTimeReachAvoid) = false

"""
    eps(spec::InfiniteTimeReachAvoid)

Return the convergence threshold of an infinite time reach-avoid specification.
"""
eps(spec::InfiniteTimeReachAvoid) = spec.eps

"""
    terminal_states(spec::InfiniteTimeReachAvoid)

Return the set of terminal states of an infinite time reach-avoid specification.
That is, the union of the reach and avoid sets.
"""
terminal_states(spec::InfiniteTimeReachAvoid) = [spec.reach; spec.avoid]

"""
    reach(spec::InfiniteTimeReachAvoid)

Return the set of target states.
"""
reach(spec::InfiniteTimeReachAvoid) = spec.reach

"""
    avoid(spec::InfiniteTimeReachAvoid)

Return the set of states to avoid.
"""
avoid(spec::InfiniteTimeReachAvoid) = spec.avoid

function checkterminal!(terminal_states, num_states)
    for j in terminal_states
        if j < 1 || j > num_states
            throw(ArgumentError("The terminal state $j is not a valid state"))
        end
    end
end

function checkdisjoint!(reach, avoid)
    if !isdisjoint(reach, avoid)
        throw(ArgumentError("The reach and avoid sets are not disjoint"))
    end
end

## Reward
"""
    AbstractReward{R <: Real}

Super type for all reward specifications.
"""
abstract type AbstractReward{R <: Real} <: Specification end

"""
    FiniteTimeReward{R <: Real, T <: Integer, VR <: AbstractVector{R}}

`FiniteTimeReward` is a specification of rewards assigned to each state at each iteration
and a discount factor. The time horizon is finite, so the discount factor is optional and 
the optimal policy will be time-varying.
"""
struct FiniteTimeReward{R <: Real, T <: Integer, VR <: AbstractVector{R}} <: AbstractReward{R}
    reward::VR
    discount::R
    time_horizon::T
end

function checkspecification!(spec::FiniteTimeReward, system::IntervalMarkovProcess)
    @assert length(reward(spec)) == num_states(system)
end

"""
    isfinitetime(spec::FiniteTimeReward)

Return `true` for FiniteTimeReward.
"""
isfinitetime(spec::FiniteTimeReward) = true

"""
    reward(spec::FiniteTimeReward)

Return the reward vector of a finite time reward optimization.
"""
reward(spec::FiniteTimeReward) = spec.reward

"""
    discount(spec::FiniteTimeReward)

Return the discount factor of a finite time reward optimization.
"""
discount(spec::FiniteTimeReward) = spec.discount

"""
    time_horizon(spec::FiniteTimeReward)

Return the time horizon of a finite time reward optimization.
"""
time_horizon(spec::FiniteTimeReward) = spec.time_horizon

"""
    InfiniteTimeReward{R <: Real, VR <: AbstractVector{R}}

`InfiniteTimeReward` is a specification of rewards assigned to each state at each iteration
and a discount factor for guaranteed convergence. The time horizon is infinite, so the optimal
policy will be stationary.
"""
struct InfiniteTimeReward{R <: Real, VR <: AbstractVector{R}} <: AbstractReward{R}
    reward::VR
    discount::R
    eps::R
end

function checkspecification!(spec::InfiniteTimeReward, system::IntervalMarkovProcess)
    @assert length(reward(spec)) == num_states(system)
end

"""
    isfinitetime(spec::InfiniteTimeReward)

Return `false` for InfiniteTimeReward.
"""
isfinitetime(spec::InfiniteTimeReward) = false

"""
    reward(spec::FiniteTimeReward)

Return the reward vector of a finite time reward optimization.
"""
reward(spec::InfiniteTimeReward) = spec.reward

"""
    discount(spec::FiniteTimeReward)

Return the discount factor of a finite time reward optimization.
"""
discount(spec::InfiniteTimeReward) = spec.discount

"""
    eps(spec::InfiniteTimeReward)

Return the convergence threshold of an infinite time reward optimization.
"""
eps(spec::InfiniteTimeReward) = spec.eps

## Problem

"""
    SatisfactionMode

When computing the satisfaction probability of a specification over an interval Markov process,
be it IMC or IMDP, the desired satisfaction probability to verify can either be `Optimistic` or
`Pessimistic`. That is, upper and lower bounds on the satisfaction probability within
the probability uncertainty.
"""
@enum SatisfactionMode Pessimistic Optimistic

"""
    Problem{S <: IntervalMarkovProcess, F <: Specification}

A problem is a tuple of an interval Markov process and a specification together with
a satisfaction mode. The satisfaction mode is either `Optimistic` or `Pessimistic`.
See [`SatisfactionMode`](@ref) for more details.

### Fields
- `system::S`: interval Markov process.
- `spec::F`: specification (either temporal logic or reachability-like).
- `mode::SatisfactionMode`: satisfaction mode (either optimistic or pessimistic). Default is pessimistic.
"""
struct Problem{S <: IntervalMarkovProcess, F <: Specification}
    system::S
    spec::F
    mode::SatisfactionMode

    function Problem(system::S, spec::F, mode::SatisfactionMode) where {S <: IntervalMarkovProcess, F <: Specification}
        checkspecification!(spec, system)
        return new{S, F}(system, spec, mode)
    end
end

function Problem(system::S, spec::F) where {S <: IntervalMarkovProcess, F <: Specification}
    return Problem(system, spec, Pessimistic)  # Default to Pessimistic
end

"""
    system(prob::Problem)

Return the system of a problem.
"""
system(prob::Problem) = prob.system

"""
    specification(prob::Problem)

Return the specification of a problem.
"""
specification(prob::Problem) = prob.spec

"""
    satisfaction_mode(prob::Problem)

Return the satisfaction mode of a problem.
"""
satisfaction_mode(prob::Problem) = prob.mode

