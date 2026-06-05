@testitem "io model" tags = [:data, :io_model] begin
    using IntervalMDP, IntervalMDP.Data, SparseArrays
    (mdp, tstates) = read_bmdp_tool_file(joinpath(@__DIR__(), "multiObj_robotIMDP.txt"))
    write_intervalmdp_jl_model(joinpath(@__DIR__(), "multiObj_robotIMDP.nc"), mdp)
    @testset "io model" begin
        mdp = read_intervalmdp_jl_model(joinpath(@__DIR__(), "multiObj_robotIMDP.nc"))
        new_path = tempname() * ".nc"
        write_intervalmdp_jl_model(new_path, mdp)
        new_mdp = read_intervalmdp_jl_model(new_path)
        rm(new_path)
        @test num_states(mdp) == num_states(new_mdp)
        marginal = (marginals(mdp))[1]
        new_marginal = (marginals(new_mdp))[1]
        as = ambiguity_sets(marginal)
        new_as = ambiguity_sets(new_marginal)
        @test source_shape(marginal) == source_shape(new_marginal)
        @test action_shape(marginal) == action_shape(new_marginal)
        @test num_target(marginal) == num_target(new_marginal)
        @test state_values(mdp) == state_values(new_mdp)
        @test action_values(mdp) == action_values(new_mdp)
        @test as.lower ≈ new_as.lower
        @test as.gap ≈ new_as.gap
    end
end

@testitem "io specification" tags = [:data, :io_specification] begin
    using IntervalMDP, IntervalMDP.Data, SparseArrays
    (mdp, tstates) = read_bmdp_tool_file(joinpath(@__DIR__(), "multiObj_robotIMDP.txt"))
    write_intervalmdp_jl_model(joinpath(@__DIR__(), "multiObj_robotIMDP.nc"), mdp)
    @testset "io specification" begin
        prop = FiniteTimeReachability([3], 10)
        pes_min_spec = Specification(prop, Pessimistic, Minimize)
        pes_max_spec = Specification(prop, Pessimistic, Maximize)
        opt_min_spec = Specification(prop, Optimistic, Minimize)
        opt_max_spec = Specification(prop, Optimistic, Maximize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, pes_min_spec)
        @test isfile(new_path)
        spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        @test satisfaction_mode(spec) == Pessimistic
        @test strategy_mode(spec) == Minimize
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, pes_max_spec)
        @test isfile(new_path)
        spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        @test satisfaction_mode(spec) == Pessimistic
        @test strategy_mode(spec) == Maximize
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, opt_min_spec)
        @test isfile(new_path)
        spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        @test satisfaction_mode(spec) == Optimistic
        @test strategy_mode(spec) == Minimize
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, opt_max_spec)
        @test isfile(new_path)
        spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        @test satisfaction_mode(spec) == Optimistic
        @test strategy_mode(spec) == Maximize
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa FiniteTimeReachability
        @test reach(new_prop) == [CartesianIndex(3)]
        @test time_horizon(new_prop) == 10
        prop = InfiniteTimeReachability([3], 1.0e-6)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa InfiniteTimeReachability
        @test reach(new_prop) == [CartesianIndex(3)]
        @test convergence_eps(new_prop) ≈ 1.0e-6
        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa FiniteTimeReachAvoid
        @test reach(new_prop) == [CartesianIndex(3)]
        @test avoid(new_prop) == [CartesianIndex(2)]
        @test time_horizon(new_prop) == 10
        prop = InfiniteTimeReachAvoid([3], [2], 1.0e-6)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa InfiniteTimeReachAvoid
        @test reach(new_prop) == [CartesianIndex(3)]
        @test avoid(new_prop) == [CartesianIndex(2)]
        @test convergence_eps(new_prop) ≈ 1.0e-6
        prop = InfiniteTimeReachAvoid([3], [2], 1.0e-6)
        spec = Specification(prop, Pessimistic, Maximize; restrict_to_initial = true)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa InfiniteTimeReachAvoid
        @test restrict_to_initial(new_spec)
        @test reach(new_prop) == [CartesianIndex(3)]
        @test avoid(new_prop) == [CartesianIndex(2)]
        @test convergence_eps(new_prop) ≈ 1.0e-6
        prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 10)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa FiniteTimeReward
        @test reward(new_prop) ≈ [1.0, 2.0, 3.0]
        @test discount(new_prop) ≈ 0.9
        @test time_horizon(new_prop) == 10
        prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1.0e-6)
        spec = Specification(prop, Pessimistic, Minimize)
        new_path = tempname() * ".json"
        write_intervalmdp_jl_spec(new_path, spec)
        @test isfile(new_path)
        new_spec = read_intervalmdp_jl_spec(new_path)
        rm(new_path)
        new_prop = system_property(new_spec)
        @test new_prop isa InfiniteTimeReward
        @test reward(new_prop) ≈ [1.0, 2.0, 3.0]
        @test discount(new_prop) ≈ 0.9
        @test convergence_eps(new_prop) ≈ 1.0e-6
    end
end

@testitem "io joint" tags = [:data, :io_joint] begin
    using IntervalMDP, IntervalMDP.Data, SparseArrays
    (mdp, tstates) = read_bmdp_tool_file(joinpath(@__DIR__(), "multiObj_robotIMDP.txt"))
    write_intervalmdp_jl_model(joinpath(@__DIR__(), "multiObj_robotIMDP.nc"), mdp)
    @testset "io joint" begin
        mdp = read_intervalmdp_jl_model(joinpath(@__DIR__(), "multiObj_robotIMDP.nc"))
        prop = FiniteTimeReachability([207], 10)
        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        new_path = tempname()
        model_path = new_path * ".nc"
        spec_path = new_path * ".json"
        write_intervalmdp_jl_model(model_path, problem)
        write_intervalmdp_jl_spec(spec_path, problem)
        @test isfile(model_path)
        @test isfile(spec_path)
        new_problem = read_intervalmdp_jl(model_path, spec_path)
        @test new_problem isa VerificationProblem
        new_problem = read_intervalmdp_jl(model_path, spec_path; control_synthesis = true)
        @test new_problem isa ControlSynthesisProblem
        rm(model_path)
        rm(spec_path)
    end
end
