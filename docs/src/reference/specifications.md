# Problem

```@docs
VerificationProblem
ControlSynthesisProblem
system
specification
strategy
Specification
system_property
Property
BasicProperty
ProductProperty
satisfaction_mode
SatisfactionMode
strategy_mode
StrategyMode
```

## DFA Reachability

```@docs
FiniteTimeDFAReachability
isfinitetime(prop::FiniteTimeDFAReachability)
terminal_states(prop::FiniteTimeDFAReachability)
reach(prop::FiniteTimeDFAReachability)
time_horizon(prop::FiniteTimeDFAReachability)

InfiniteTimeDFAReachability
isfinitetime(prop::InfiniteTimeDFAReachability)
terminal_states(prop::InfiniteTimeDFAReachability)
reach(prop::InfiniteTimeDFAReachability)
convergence_eps(prop::InfiniteTimeDFAReachability)
```

## Reachability

```@docs
FiniteTimeReachability
isfinitetime(prop::FiniteTimeReachability)
terminal_states(prop::FiniteTimeReachability)
reach(prop::FiniteTimeReachability)
time_horizon(prop::FiniteTimeReachability)

InfiniteTimeReachability
isfinitetime(prop::InfiniteTimeReachability)
terminal_states(prop::InfiniteTimeReachability)
reach(prop::InfiniteTimeReachability)
convergence_eps(prop::InfiniteTimeReachability)

ExactTimeReachability
isfinitetime(prop::ExactTimeReachability)
terminal_states(prop::ExactTimeReachability)
reach(prop::ExactTimeReachability)
time_horizon(prop::ExactTimeReachability)
```

## Reach-avoid

```@docs
FiniteTimeReachAvoid
isfinitetime(prop::FiniteTimeReachAvoid)
terminal_states(prop::FiniteTimeReachAvoid)
reach(prop::FiniteTimeReachAvoid)
avoid(prop::FiniteTimeReachAvoid)
time_horizon(prop::FiniteTimeReachAvoid)

InfiniteTimeReachAvoid
isfinitetime(prop::InfiniteTimeReachAvoid)
terminal_states(prop::InfiniteTimeReachAvoid)
reach(prop::InfiniteTimeReachAvoid)
avoid(prop::InfiniteTimeReachAvoid)
convergence_eps(prop::InfiniteTimeReachAvoid)

ExactTimeReachAvoid
isfinitetime(prop::ExactTimeReachAvoid)
terminal_states(prop::ExactTimeReachAvoid)
reach(prop::ExactTimeReachAvoid)
avoid(prop::ExactTimeReachAvoid)
time_horizon(prop::ExactTimeReachAvoid)
```

## Safety

```@docs
FiniteTimeSafety
isfinitetime(prop::FiniteTimeSafety)
terminal_states(prop::FiniteTimeSafety)
avoid(prop::FiniteTimeSafety)
time_horizon(prop::FiniteTimeSafety)

InfiniteTimeSafety
isfinitetime(prop::InfiniteTimeSafety)
terminal_states(prop::InfiniteTimeSafety)
avoid(prop::InfiniteTimeSafety)
convergence_eps(prop::InfiniteTimeSafety)
```

## Reward specification

```@docs
FiniteTimeReward
isfinitetime(prop::FiniteTimeReward)
reward(prop::FiniteTimeReward)
discount(prop::FiniteTimeReward)
time_horizon(prop::FiniteTimeReward)

InfiniteTimeReward
isfinitetime(prop::InfiniteTimeReward)
reward(prop::InfiniteTimeReward)
discount(prop::InfiniteTimeReward)
convergence_eps(prop::InfiniteTimeReward)
```

## Hitting time

```@docs
ExpectedExitTime
isfinitetime(prop::ExpectedExitTime)
terminal_states(prop::ExpectedExitTime)
avoid(prop::ExpectedExitTime)
convergence_eps(prop::ExpectedExitTime)
```