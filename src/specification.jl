## Specification types

"""
    Specification

Super type for all system specficiations
"""
abstract type Specification end


# Temporal logics

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


# Reachability

"""
    AbstractReachability

Super type for all reachability-like specifications.
"""
abstract type AbstractReachability <: Specification end

"""
    FiniteTimeReachability{T <: Integer}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, if ``T`` is the set of target states and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T)``.
"""
struct FiniteTimeReachability{T <: Integer} <: AbstractReachability
    terminal_states::Vector{T}
    time_horizon::Any
end

function FiniteTimeReachability(
    terminal_states::Vector{T},
    num_states::T,
    time_horizon,
) where {T <: Integer}
    checkterminal!(terminal_states, num_states)
    return FiniteTimeReachability(terminal_states, time_horizon)
end

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
    InfiniteTimeReachability{R <: Real, T <: Integer} 
 
`InfiniteTimeReachability` is similar to [`FiniteTimeReachability`](@ref) except that the time horizon is infinite.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `eps`.
"""
struct InfiniteTimeReachability{R <: Real, T <: Integer} <: AbstractReachability
    terminal_states::Vector{T}
    eps::R
end

function InfiniteTimeReachability(
    terminal_states::Vector{T},
    num_states::T,
    eps::R,
) where {R <: Real, T <: Integer}
    checkterminal!(terminal_states, num_states)
    return InfiniteTimeReachability(terminal_states, eps)
end

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

"""
    FiniteTimeReachAvoid{T <: Integer}

Finite time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, if ``T`` is the set of target states, ``A`` is the set of states to avoid, and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T and ∀k' = 0…k, s_k' ∉ A)``.
"""
struct FiniteTimeReachAvoid{T <: Integer} <: AbstractReachability
    reach::Vector{T}
    avoid::Vector{T}
    time_horizon::Any
end

function FiniteTimeReachAvoid(
    reach::Vector{T},
    avoid::Vector{T},
    num_states::T,
    time_horizon,
) where {T <: Integer}
    checkterminal!(reach, num_states)
    checkterminal!(avoid, num_states)
    checkdisjoint!(reach, avoid)
    return FiniteTimeReachAvoid(reach, avoid, time_horizon)
end

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
    InfiniteTimeReachAvoid{R <: Real, T <: Integer}

`InfiniteTimeReachAvoid` is similar to [`FiniteTimeReachAvoid`](@ref) except that the time horizon is infinite.
"""
struct InfiniteTimeReachAvoid{R <: Real, T <: Integer} <: AbstractReachability
    reach::Vector{T}
    avoid::Vector{T}
    eps::R
end

function InfiniteTimeReachAvoid(
    reach::Vector{T},
    avoid::Vector{T},
    num_states::T,
    eps::R,
) where {R <: Real, T <: Integer}
    checkterminal!(reach, num_states)
    checkterminal!(avoid, num_states)
    checkdisjoint!(reach, avoid)
    return InfiniteTimeReachAvoid(reach, avoid, eps)
end

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

# Problem

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

# TODO: Add `checkspecification!(spec, system)` to ensure that the specification is valid for the system.