# Problem

```@docs
VerificationProblem
ControlSynthesisProblem
system
specification
strategy(prob::VerificationProblem)
Specification
system_property
satisfaction_mode
SatisfactionMode
strategy_mode
StrategyMode
```

## DFA Reachability

```@docs
FiniteTimeDFAReachability
reach(prop::FiniteTimeDFAReachability)
time_horizon(prop::FiniteTimeDFAReachability)

InfiniteTimeDFAReachability
reach(prop::InfiniteTimeDFAReachability)
convergence_eps(prop::InfiniteTimeDFAReachability)
```

## Reachability

```@docs
FiniteTimeReachability
reach(prop::FiniteTimeReachability)
time_horizon(prop::FiniteTimeReachability)

InfiniteTimeReachability
reach(prop::InfiniteTimeReachability)
convergence_eps(prop::InfiniteTimeReachability)

ExactTimeReachability
reach(prop::ExactTimeReachability)
time_horizon(prop::ExactTimeReachability)
```

## Reach-avoid

```@docs
FiniteTimeReachAvoid
reach(prop::FiniteTimeReachAvoid)
avoid(prop::FiniteTimeReachAvoid)
time_horizon(prop::FiniteTimeReachAvoid)

InfiniteTimeReachAvoid
reach(prop::InfiniteTimeReachAvoid)
avoid(prop::InfiniteTimeReachAvoid)
convergence_eps(prop::InfiniteTimeReachAvoid)

ExactTimeReachAvoid
reach(prop::ExactTimeReachAvoid)
avoid(prop::ExactTimeReachAvoid)
time_horizon(prop::ExactTimeReachAvoid)
```

## Safety

```@docs
FiniteTimeSafety
avoid(prop::FiniteTimeSafety)
time_horizon(prop::FiniteTimeSafety)

InfiniteTimeSafety
avoid(prop::InfiniteTimeSafety)
convergence_eps(prop::InfiniteTimeSafety)
```

## Reward specification

```@docs
FiniteTimeReward
reward(prop::FiniteTimeReward)
discount(prop::FiniteTimeReward)
time_horizon(prop::FiniteTimeReward)

InfiniteTimeReward
reward(prop::InfiniteTimeReward)
discount(prop::InfiniteTimeReward)
convergence_eps(prop::InfiniteTimeReward)
```

## Hitting time

```@docs
ExpectedExitTime
avoid(prop::ExpectedExitTime)
convergence_eps(prop::ExpectedExitTime)
```