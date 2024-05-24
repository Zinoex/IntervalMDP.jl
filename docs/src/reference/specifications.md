# Problem

```@docs
Problem
system
specification
Specification
system_property
Property
satisfaction_mode
SatisfactionMode
strategy_mode
StrategyMode
```

## Temporal logic

```@docs
LTLFormula
isfinitetime(spec::LTLFormula)
LTLfFormula
isfinitetime(spec::LTLfFormula)
time_horizon(spec::LTLfFormula)
PCTLFormula
```

## Reachability

```@docs
AbstractReachability

FiniteTimeReachability
isfinitetime(spec::FiniteTimeReachability)
terminal_states(spec::FiniteTimeReachability)
reach(spec::FiniteTimeReachability)
time_horizon(spec::FiniteTimeReachability)

InfiniteTimeReachability
isfinitetime(spec::InfiniteTimeReachability)
terminal_states(spec::InfiniteTimeReachability)
reach(spec::InfiniteTimeReachability)
convergence_eps(spec::InfiniteTimeReachability)
```

## Reach-avoid

```@docs
AbstractReachAvoid

FiniteTimeReachAvoid
isfinitetime(spec::FiniteTimeReachAvoid)
terminal_states(spec::FiniteTimeReachAvoid)
reach(spec::FiniteTimeReachAvoid)
avoid(spec::FiniteTimeReachAvoid)
time_horizon(spec::FiniteTimeReachAvoid)

InfiniteTimeReachAvoid
isfinitetime(spec::InfiniteTimeReachAvoid)
terminal_states(spec::InfiniteTimeReachAvoid)
reach(spec::InfiniteTimeReachAvoid)
avoid(spec::InfiniteTimeReachAvoid)
convergence_eps(spec::InfiniteTimeReachAvoid)
```

## Reward specification

```@docs
AbstractReward

FiniteTimeReward
isfinitetime(spec::FiniteTimeReward)
reward(spec::FiniteTimeReward)
discount(spec::FiniteTimeReward)
time_horizon(spec::FiniteTimeReward)

InfiniteTimeReward
isfinitetime(spec::InfiniteTimeReward)
reward(spec::InfiniteTimeReward)
discount(spec::InfiniteTimeReward)
convergence_eps(spec::InfiniteTimeReward)
```