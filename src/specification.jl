## Specification types
abstract type Specification end

# Temporal logics
abstract type AbstractTemporalLogic <: Specification end
struct LTLFormula <: AbstractTemporalLogic
    formula::String
end

struct LTLfFormula <: AbstractTemporalLogic
    formula::String
    time_horizon::Int32
end
time_horizon(spec::LTLfFormula) = spec.time_horizon

struct PCTLFormula <: AbstractTemporalLogic
    formula::String
end

# Reachability
abstract type AbstractReachability <: Specification end
struct FiniteTimeReachability <: AbstractReachability
    terminal_states::Vector{Int32}
    time_horizon::Int32
end

function FiniteTimeReachability(terminal_states::Vector{Int32}, num_states::Int32, time_horizon::Int32)
    checkterminal!(terminal_states, num_states)
    return FiniteTimeReachability(terminal_states, time_horizon)
end

time_horizon(spec::FiniteTimeReachability) = spec.time_horizon
terminal_states(spec::FiniteTimeReachability) = spec.terminal_states

struct InfiniteTimeReachability{R} <: AbstractReachability
    eps::R
    terminal_states::Vector{Int32}
end

function InfiniteTimeReachability(terminal_states::Vector{Int32}, num_states::Int32, eps)
    checkterminal!(terminal_states, num_states)
    return InfiniteTimeReachability(terminal_states, eps)
end

eps(spec::InfiniteTimeReachability) = spec.eps
terminal_states(spec::InfiniteTimeReachability) = spec.terminal_states

function checkterminal!(terminal_states, num_states)
    for j in terminal_states
        if j < 1 || j > num_states
            throw(ArgumentError("The terminal state $j is not a valid state"))
        end
    end
end

# Problem
@enum SatisfactionMode Pessimistic Optimistic
struct Problem{S <: System, F <: Specification}
    system::S
    spec::F
    mode::SatisfactionMode
end

system(prob::Problem) = prob.system
specification(prob::Problem) = prob.spec
satisfaction_mode(prob::Problem) = prob.mode
