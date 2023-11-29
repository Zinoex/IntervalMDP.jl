# Specifications

```@docs
Specification
```

## Temporal logic

```@docs
IMDP.AbstractTemporalLogic

LTLFormula
LTLfFormula
time_horizon(spec::LTLfFormula)
PCTLFormula
```

## Reachability

```@docs
AbstractReachability

FiniteTimeReachability
terminal_states(spec::FiniteTimeReachability)
reach(spec::FiniteTimeReachability)
time_horizon(spec::FiniteTimeReachability)

InfiniteTimeReachability
terminal_states(spec::InfiniteTimeReachability)
reach(spec::InfiniteTimeReachability)
IMDP.eps(spec::InfiniteTimeReachability)
```

## Reach-avoid

```@docs
AbstractReachAvoid

FiniteTimeReachAvoid
terminal_states(spec::FiniteTimeReachAvoid)
reach(spec::FiniteTimeReachAvoid)
avoid(spec::FiniteTimeReachAvoid)
time_horizon(spec::FiniteTimeReachAvoid)

InfiniteTimeReachAvoid
terminal_states(spec::InfiniteTimeReachAvoid)
reach(spec::InfiniteTimeReachAvoid)
avoid(spec::InfiniteTimeReachAvoid)
IMDP.eps(spec::InfiniteTimeReachAvoid)
```

## Problem
```@docs
SatisfactionMode
Problem
system
specification
satisfaction_mode
```