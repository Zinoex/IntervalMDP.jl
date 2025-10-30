using Revise, Test
using IntervalMDP

@testset "getters" begin
    @testset "DFA reachability" begin
        prop = FiniteTimeDFAReachability([3], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reach(prop) == [3]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeDFAReachability", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reach states: Int32[3]", str)

        prop = InfiniteTimeDFAReachability([3], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test reach(prop) == [3]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeDFAReachability", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Reach states: Int32[3]", str)
    end

    @testset "DFA safety" begin
        prop = FiniteTimeDFASafety([3], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test avoid(prop) == [3]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeDFASafety", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Avoid states: Int32[3]", str)

        prop = InfiniteTimeDFASafety([3], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test avoid(prop) == [3]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeDFASafety", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Avoid states: Int32[3]", str)
    end

    @testset "reachability" begin
        prop = FiniteTimeReachability([3], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reach(prop) == [CartesianIndex(3)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeReachability", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)

        prop = InfiniteTimeReachability([3], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test reach(prop) == [CartesianIndex(3)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeReachability", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)

        prop = ExactTimeReachability([3], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reach(prop) == [CartesianIndex(3)]
        
        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("ExactTimeReachability", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)
    end

    @testset "reach-avoid" begin
        prop = FiniteTimeReachAvoid([3], [4], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reach(prop) == [CartesianIndex(3)]
        @test avoid(prop) == [CartesianIndex(4)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeReachAvoid", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)
        @test occursin("Avoid states: CartesianIndex{1}[CartesianIndex(4,)]", str)

        prop = InfiniteTimeReachAvoid([3], [4], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test reach(prop) == [CartesianIndex(3)]
        @test avoid(prop) == [CartesianIndex(4)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeReachAvoid", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)
        @test occursin("Avoid states: CartesianIndex{1}[CartesianIndex(4,)]", str)

        prop = ExactTimeReachAvoid([3], [4], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reach(prop) == [CartesianIndex(3)]
        @test avoid(prop) == [CartesianIndex(4)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("ExactTimeReachAvoid", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reach states: CartesianIndex{1}[CartesianIndex(3,)]", str)
        @test occursin("Avoid states: CartesianIndex{1}[CartesianIndex(4,)]", str)
    end

    @testset "safety" begin
        prop = FiniteTimeSafety([3], 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test avoid(prop) == [CartesianIndex(3)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeSafety", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Avoid states: CartesianIndex{1}[CartesianIndex(3,)]", str)

        prop = InfiniteTimeSafety([3], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test avoid(prop) == [CartesianIndex(3)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeSafety", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Avoid states: CartesianIndex{1}[CartesianIndex(3,)]", str)
    end

    @testset "reward" begin
        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 10)
        @test IntervalMDP.isfinitetime(prop)
        @test time_horizon(prop) == 10

        @test reward(prop) == [1.0, 2.0, 3.0]
        @test discount(prop) == 0.9

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("FiniteTimeReward", str)
        @test occursin("Time horizon: 10", str)
        @test occursin("Reward storage: Float64, (3,)", str)
        @test occursin("Discount factor: 0.9", str)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test reward(prop) == [1.0, 2.0, 3.0]
        @test discount(prop) == 0.9

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("InfiniteTimeReward", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
        @test occursin("Reward storage: Float64, (3,)", str)
        @test occursin("Discount factor: 0.9", str)
    end

    @testset "expected exit time" begin
        prop = ExpectedExitTime([3], 1e-6)
        @test !IntervalMDP.isfinitetime(prop)
        @test convergence_eps(prop) == 1e-6

        @test avoid(prop) == [CartesianIndex(3)]

        io = IOBuffer()
        show(io, MIME("text/plain"), prop)
        str = String(take!(io))
        @test occursin("ExpectedExitTime", str)
        @test occursin("Convergence threshold: 1.0e-6", str)
    end
end

@testset "specification" begin
    prop = FiniteTimeDFAReachability([3], 10)

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

    io = IOBuffer()
    show(io, MIME("text/plain"), spec)
    str = String(take!(io))
    @test occursin("Specification", str)
    @test occursin("Satisfaction mode: Optimistic", str)
    @test occursin("Strategy mode: Minimize", str)
    @test occursin("Property: FiniteTimeDFAReachability", str)
end

##########
# Errors #
##########
@testset "errors" begin
    prob = IntervalAmbiguitySets(;
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
    tv_strat = TimeVaryingStrategy([Tuple{Int32}[(1,), (1,)]])

    # Product model - just simple reachability
    delta = TransitionFunction(Int32[
        1 2
        2 2
    ])
    istate = Int32(1)
    atomic_props = ["reach"]
    dfa = DFA(delta, istate, atomic_props)

    labelling = DeterministicLabelling(Int32[1, 1, 2])

    prod_proc = ProductProcess(mc, dfa, labelling)
    tv_prod_strat = TimeVaryingStrategy([Tuple{Int32}[
        (1,) (1,)
        (1,) (1,)
        (1,) (1,)
    ]])

    # Time horizon must be a positive integer
    @testset "time horizon" begin
        prop = FiniteTimeDFAReachability([2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = FiniteTimeDFAReachability([2], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = FiniteTimeDFAReachability([2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeDFASafety([2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = FiniteTimeDFASafety([2], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = FiniteTimeDFASafety([2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReachability([3], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReachAvoid([3], [2], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExactTimeReachability([3], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExactTimeReachability([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExactTimeReachAvoid([3], [2], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExactTimeReachAvoid([3], [2], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeSafety([3], -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeSafety([3], 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, -1)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec, tv_strat)
    end

    # Time horizon must be equal to the time length of the time-varying interval Markov process
    @testset "time horizon/time-varying strategy" begin
        prop = FiniteTimeDFAReachability([2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeDFAReachability([2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeDFASafety([2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeDFASafety([2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = FiniteTimeReachability([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReachability([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReachAvoid([3], [2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachability([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = ExactTimeReachAvoid([3], [2], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeSafety([3], 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 2)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 4)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)
    end

    # Convergence threshold must be a positive number
    @testset "convergence epsilon" begin
        prop = InfiniteTimeDFAReachability([2], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = InfiniteTimeDFAReachability([2], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = InfiniteTimeDFASafety([2], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = InfiniteTimeDFASafety([2], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(prod_proc, spec)

        prop = InfiniteTimeReachability([3], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReachability([3], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReachAvoid([3], [2], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReachAvoid([3], [2], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeSafety([3], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeSafety([3], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExpectedExitTime([3], 0.0)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = ExpectedExitTime([3], -1e-3)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)
    end

    # Infinite time properties are not supported for time-varying interval Markov processes
    @testset "infinite time property/time-varying controller" begin
        prop = InfiniteTimeDFAReachability([2], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)
    
        prop = InfiniteTimeDFASafety([2], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec, tv_prod_strat)

        prop = InfiniteTimeReachability([3], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = InfiniteTimeSafety([3], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)

        prop = ExpectedExitTime([3], 1e-6)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec, tv_strat)
    end

    # Specification state errors
    @testset "specification state errors" begin
        @testset "finite time" begin
            @testset "DFA reachability" begin
                prop = FiniteTimeDFAReachability([3], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)

                prop = FiniteTimeDFAReachability([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)
            end

            @testset "DFA safety" begin
                prop = FiniteTimeDFASafety([3], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)

                prop = FiniteTimeDFASafety([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)
            end

            @testset "reachability" begin
                prop = FiniteTimeReachability([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachability([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachability([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = FiniteTimeReachAvoid([4], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachAvoid([0], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeReachAvoid([(3, 2)], [(2, 3)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)

                prop = FiniteTimeReachAvoid([2], [2], 10) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError VerificationProblem(mc, spec)
            end

            @testset "safety" begin
                prop = FiniteTimeSafety([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeSafety([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = FiniteTimeSafety([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end
        end

        @testset "exact time" begin
            @testset "reachability" begin
                prop = ExactTimeReachability([4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachability([0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachability([(3, 2)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = ExactTimeReachAvoid([4], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachAvoid([0], [2], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachAvoid([2], [4], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachAvoid([2], [0], 10) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExactTimeReachAvoid([(3, 2)], [(2, 3)], 10) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)

                prop = ExactTimeReachAvoid([2], [2], 10) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError VerificationProblem(mc, spec)
            end
        end

        @testset "infinite time" begin
            @testset "DFA reachability" begin
                prop = InfiniteTimeDFAReachability([3], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)

                prop = InfiniteTimeDFAReachability([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)
            end

            @testset "DFA safety" begin
                prop = InfiniteTimeDFASafety([3], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)

                prop = InfiniteTimeDFASafety([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(prod_proc, spec)
            end

            @testset "reachability" begin
                prop = InfiniteTimeReachability([4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachability([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachability([(3, 2)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end

            @testset "reach/avoid" begin
                prop = InfiniteTimeReachAvoid([4], [2], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachAvoid([0], [2], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachAvoid([(3, 2)], [(2, 3)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeReachAvoid([2], [2], 1e-6) # not disjoint
                spec = Specification(prop)
                @test_throws DomainError VerificationProblem(mc, spec)
            end

            @testset "safety" begin
                prop = InfiniteTimeSafety([4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeSafety([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = InfiniteTimeSafety([(3, 2)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end

            @testset "expected exit time" begin
                prop = ExpectedExitTime([4], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExpectedExitTime([0], 1e-6) # out-of-bounds
                spec = Specification(prop)
                @test_throws InvalidStateError VerificationProblem(mc, spec)

                prop = ExpectedExitTime([(3, 2)], 1e-6) # incorrect dimension
                spec = Specification(prop)
                @test_throws StateDimensionMismatch VerificationProblem(mc, spec)
            end
        end
    end

    @testset "reward discount" begin
        # Reward discount factor must be in the range (0, ∞) for finite time properties
        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.0, 10)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = FiniteTimeReward([1.0, 2.0, 3.0], -1.0, 10)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        # Reward discount factor must be in the range (0, 1) for infinite time properties
        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 1.0, 1e-6)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.0, 1e-6)
        spec = Specification(prop)
        @test_throws DomainError VerificationProblem(mc, spec)
    end

    # Reward shape
    @testset "reward shape" begin
        prop = FiniteTimeReward([1.0, 2.0], 0.9, 10)
        spec = Specification(prop)
        @test_throws DimensionMismatch VerificationProblem(mc, spec)

        prop = InfiniteTimeReward([1.0, 2.0], 0.9, 1e-6)
        spec = Specification(prop)
        @test_throws DimensionMismatch VerificationProblem(mc, spec)
    end

    @testset "incompatible model and specification" begin
        prop = FiniteTimeDFAReachability([2], 10)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(mc, spec)

        prop = FiniteTimeReachability([2], 10)
        spec = Specification(prop)
        @test_throws ArgumentError VerificationProblem(prod_proc, spec)
    end
end
