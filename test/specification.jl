# LTL
prop = LTLFormula("G !avoid")
@test !isfinitetime(prop)

prop = LTLfFormula("G !avoid", 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

# Reachability
prop = FiniteTimeReachability([3], 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

@test reach(prop) == [CartesianIndex(3)]
@test terminal_states(prop) == [CartesianIndex(3)]

prop = InfiniteTimeReachability([3], 1e-6)
@test !isfinitetime(prop)
@test convergence_eps(prop) == 1e-6

@test reach(prop) == [CartesianIndex(3)]
@test terminal_states(prop) == [CartesianIndex(3)]

# Reach-avoid
prop = FiniteTimeReachAvoid([3], [4], 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

@test reach(prop) == [CartesianIndex(3)]
@test avoid(prop) == [CartesianIndex(4)]
@test issetequal(terminal_states(prop), [CartesianIndex(3), CartesianIndex(4)])

prop = InfiniteTimeReachAvoid([3], [4], 1e-6)
@test !isfinitetime(prop)
@test convergence_eps(prop) == 1e-6

@test reach(prop) == [CartesianIndex(3)]
@test avoid(prop) == [CartesianIndex(4)]
@test issetequal(terminal_states(prop), [CartesianIndex(3), CartesianIndex(4)])

# Reward
prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

@test reward(prop) == [1.0, 2.0, 3.0]
@test discount(prop) == 0.9

prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
@test !isfinitetime(prop)
@test convergence_eps(prop) == 1e-6

@test reward(prop) == [1.0, 2.0, 3.0]
@test discount(prop) == 0.9

# Default
spec = Specification(prop)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize
@test system_property(spec) == prop

# IMC
spec = Specification(prop, Optimistic)
@test satisfaction_mode(spec) == Optimistic
@test system_property(spec) == prop

# IMDP
spec = Specification(prop, Optimistic, Minimize)
@test satisfaction_mode(spec) == Optimistic
@test strategy_mode(spec) == Minimize
@test system_property(spec) == prop
