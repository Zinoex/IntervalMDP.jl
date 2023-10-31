## Specification types
abstract type Specification end

# Temporal logics
abstract type AbstractTemporalLogic <: Specification end
struct LTLFormula <: AbstractTemporalLogic
    formula::String
end

struct LTLfFormula{T <: Integer} <: AbstractTemporalLogic
    formula::String
    time_horizon::T
end
time_horizon(spec::LTLfFormula) = spec.time_horizon

struct PCTLFormula <: AbstractTemporalLogic
    formula::String
end

# Reachability
abstract type AbstractReachability <: Specification end
struct FiniteTimeReachability{T <: Integer} <: AbstractReachability
    terminal_states::Vector{T}
    time_horizon
end

function FiniteTimeReachability(
    terminal_states::Vector{T},
    num_states::T,
    time_horizon,
) where {T <: Integer}
    checkterminal!(terminal_states, num_states)
    return FiniteTimeReachability(terminal_states, time_horizon)
end

time_horizon(spec::FiniteTimeReachability) = spec.time_horizon
terminal_states(spec::FiniteTimeReachability) = spec.terminal_states
reach(spec::FiniteTimeReachability) = spec.terminal_states

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

eps(spec::InfiniteTimeReachability) = spec.eps
terminal_states(spec::InfiniteTimeReachability) = spec.terminal_states
reach(spec::InfiniteTimeReachability) = spec.terminal_states

struct FiniteTimeReachAvoid{T <: Integer} <: AbstractReachability
    reach::Vector{T}
    avoid::Vector{T}
    time_horizon
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

time_horizon(spec::FiniteTimeReachAvoid) = spec.time_horizon
terminal_states(spec::FiniteTimeReachAvoid) = [spec.reach; spec.avoid]
reach(spec::FiniteTimeReachAvoid) = spec.reach
avoid(spec::FiniteTimeReachAvoid) = spec.avoid

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

eps(spec::InfiniteTimeReachAvoid) = spec.eps
terminal_states(spec::InfiniteTimeReachAvoid) = [spec.reach; spec.avoid]
reach(spec::InfiniteTimeReachAvoid) = spec.reach
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
@enum SatisfactionMode Pessimistic Optimistic
struct Problem{S <: System, F <: Specification}
    system::S
    spec::F
    mode::SatisfactionMode
end

function Problem(system::S, spec::F) where {S <: System, F <: Specification}
    return Problem(system, spec, Pessimistic)  # Default to Pessimistic
end

system(prob::Problem) = prob.system
specification(prob::Problem) = prob.spec
satisfaction_mode(prob::Problem) = prob.mode
