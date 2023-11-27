# Specifications

```@docs
Specification
```

## Temporal logic

```@docs
IMDP.AbstractTemporalLogic

LTLFormula
IMDP.isfinitetime(spec::LTLFormula)
LTLfFormula
IMDP.isfinitetime(spec::LTLfFormula)
time_horizon(spec::LTLfFormula)
PCTLFormula
```

## Reachability

```@docs
AbstractReachability

FiniteTimeReachability
IMDP.isfinitetime(spec::FiniteTimeReachability)
terminal_states(spec::FiniteTimeReachability)
reach(spec::FiniteTimeReachability)
time_horizon(spec::FiniteTimeReachability)

InfiniteTimeReachability
IMDP.isfinitetime(spec::InfiniteTimeReachability)
terminal_states(spec::InfiniteTimeReachability)
reach(spec::InfiniteTimeReachability)
IMDP.eps(spec::InfiniteTimeReachability)
```

## Reach-avoid

```@docs
FiniteTimeReachAvoid
IMDP.isfinitetime(spec::FiniteTimeReachAvoid)
terminal_states(spec::FiniteTimeReachAvoid)
reach(spec::FiniteTimeReachAvoid)
avoid(spec::FiniteTimeReachAvoid)
time_horizon(spec::FiniteTimeReachAvoid)

InfiniteTimeReachAvoid
IMDP.isfinitetime(spec::InfiniteTimeReachAvoid)
terminal_states(spec::InfiniteTimeReachAvoid)
reach(spec::InfiniteTimeReachAvoid)
avoid(spec::InfiniteTimeReachAvoid)
IMDP.eps(spec::InfiniteTimeReachAvoid)
```

## Reward specification

```@docs
AbstractReward

FiniteTimeReward
IMDP.isfinitetime(spec::FiniteTimeReward)
reward(spec::FiniteTimeReward)
discount(spec::FiniteTimeReward)
time_horizon(spec::FiniteTimeReward)

InfiniteTimeReward
IMDP.isfinitetime(spec::InfiniteTimeReward)
reward(spec::InfiniteTimeReward)
discount(spec::InfiniteTimeReward)
IMDP.eps(spec::InfiniteTimeReward)
```

## Problem
```@docs
SatisfactionMode
Problem
system
specification
satisfaction_mode
```