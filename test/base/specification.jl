using Revise, Test
using IntervalMDP

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

prop = ExactTimeReachability([3], 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

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

prop = ExactTimeReachAvoid([3], [4], 10)
@test isfinitetime(prop)
@test time_horizon(prop) == 10

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

# Convenience
@test !isoptimistic(Pessimistic)
@test isoptimistic(Optimistic)
@test ispessimistic(Pessimistic)
@test !ispessimistic(Optimistic)

@test isoptimistic(!Pessimistic)
@test ispessimistic(!Optimistic)

@test ismaximize(Maximize)
@test !ismaximize(Minimize)
@test !isminimize(Maximize)
@test isminimize(Minimize)

@test isminimize(!Maximize)
@test ismaximize(!Minimize)

# IMC
spec = Specification(prop, Optimistic)
@test satisfaction_mode(spec) == Optimistic
@test system_property(spec) == prop

# IMDP
spec = Specification(prop, Optimistic, Minimize)
@test satisfaction_mode(spec) == Optimistic
@test strategy_mode(spec) == Minimize
@test system_property(spec) == prop

##########
# Errors #
##########
@testset "errors" begin
    prob = IntervalProbabilities(;
        lower = [
            0.0 0.5 0.0
            0.1 0.3 0.0
            0.2 0.1 1.0
        ],
        upper = [
            0.5 0.7 0.0
            0.6 0.5 0.0
            0.7 0.3 1.0
        ],
    )
    mc = IntervalMarkovChain(prob)
    tv_strat = TimeVaryingStrategy([Int32[1, 1, 1]])

    # Time horizon must be a positive integer
    @testset "time horizon" begin
        prop = FiniteTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReachability([3], -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReachAvoid([3], [2], -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = ExactTimeReachability([3], -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = ExactTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = ExactTimeReachAvoid([3], [2], -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = ExactTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeSafety([3], -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeSafety([3], 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, -1)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec, tv_strat)
    end

    # Time horizon must be equal to the time length of the time-varying interval Markov process
    @testset "time horizon/time-varying strategy" begin
        prop = FiniteTimeReachability([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReachability([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 2)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 4)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)
    end

    # Convergence epsilon must be a positive number
    @testset "convergence epsilon" begin
        prop = InfiniteTimeReachability([3], 0.0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReachability([3], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReachAvoid([3], [2], 0.0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReachAvoid([3], [2], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeSafety([3], 0.0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeSafety([3], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0.0)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, -1e-3)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)
    end

    # Infinite time properties are not supported for time-varying interval Markov processes
    @testset "infinite time property/time-varying controller" begin
        prop = InfiniteTimeReachability([3], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = InfiniteTimeSafety([3], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError Problem(mc, spec, tv_strat)
    end

    # Specification state errors
    @testset "specification state errors" begin
        @testset "finite time" begin
            @testset "reachability" begin
                prop = FiniteTimeReachability([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachability([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachability([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = FiniteTimeReachAvoid([4], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachAvoid([0], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeReachAvoid([(3, 2)], [(2, 3)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [2], 10) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError Problem(mc, spec)
            end

            @testset "safety" begin
                prop = FiniteTimeSafety([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeSafety([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = FiniteTimeSafety([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)
            end
        end

        @testset "exact time" begin
            @testset "reachability" begin
                prop = ExactTimeReachability([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachability([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachability([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = ExactTimeReachAvoid([4], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachAvoid([0], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachAvoid([2], [4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachAvoid([2], [0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = ExactTimeReachAvoid([(3, 2)], [(2, 3)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)

                prop = ExactTimeReachAvoid([2], [2], 10) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError Problem(mc, spec)
            end
        end

        @testset "infinite time" begin
            @testset "reachability" begin
                prop = InfiniteTimeReachability([4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachability([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachability([(3, 2)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = InfiniteTimeReachAvoid([4], [2], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachAvoid([0], [2], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachAvoid([(3, 2)], [(2, 3)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [2], 1e-6) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError Problem(mc, spec)
            end

            @testset "safety" begin
                prop = InfiniteTimeSafety([4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeSafety([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError Problem(mc, spec)

                prop = InfiniteTimeSafety([(3, 2)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch Problem(mc, spec)
            end
        end
    end

    @testset "reward discount" begin
        # Reward discount factor must be in the range (0, ∞) for finite time properties
        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.0, 10)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], -1.0, 10)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        # Reward discount factor must be in the range (0, 1) for infinite time properties
        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 1.0, 1e-6)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.0, 1e-6)
        spec = Specification(prop)
        @test_throws DomainError Problem(mc, spec)
    end

    # Reward shape
    @testset "reward shape" begin
        prop = FiniteTimeReward([1.0, 2.0], 0.9, 10)
        spec = Specification(prop)
        @test_throws DimensionMismatch Problem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0], 0.9, 1e-6)
        spec = Specification(prop)
        @test_throws DimensionMismatch Problem(mc, spec)
    end
end
