@testitem "data/bmdp_tool: write/read model,tstates" begin
    using IntervalMDP.Data
    using SparseArrays

    # Read MDP
    mdp, tstates = read_bmdp_tool_file("multiObj_robotIMDP.txt")

    marginal = marginals(mdp)[1]
    as = ambiguity_sets(marginal)

    # Write model
    new_path = tempname() * ".txt"
    write_bmdp_tool_file(new_path, mdp, tstates)

    # Check the file is there
    @test isfile(new_path)

    # Read new file and check that the models are the same
    new_mdp, new_tstates = read_bmdp_tool_file(new_path)
    rm(new_path)

    @test num_states(mdp) == num_states(new_mdp)

    new_marginal = marginals(new_mdp)[1]
    new_as = ambiguity_sets(new_marginal)

    @test source_shape(marginal) == source_shape(new_marginal)
    @test action_shape(marginal) == action_shape(new_marginal)
    @test num_target(marginal) == num_target(new_marginal)
    @test state_values(mdp) == state_values(new_mdp)
    @test action_values(mdp) == action_values(new_mdp)

    @test as.lower ≈ new_as.lower
    @test as.gap ≈ new_as.gap

    @test tstates == new_tstates
end

@testitem "data/bmdp_tool: write/read problem" begin
    using IntervalMDP.Data
    using SparseArrays

    # Read MDP
    mdp, tstates = read_bmdp_tool_file("multiObj_robotIMDP.txt")

    marginal = marginals(mdp)[1]
    as = ambiguity_sets(marginal)

    # Write problem
    tstates = [CartesianIndex(207)]
    prop = FiniteTimeReachability(tstates, 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)

    new_path = tempname() * ".txt"
    write_bmdp_tool_file(new_path, problem)

    # Check the file is there
    @test isfile(new_path)

    # Read new file and check that the models represent the same system
    new_mdp, new_tstates = read_bmdp_tool_file(new_path)

    @test num_states(mdp) == num_states(new_mdp)

    new_marginal = marginals(new_mdp)[1]
    new_as = ambiguity_sets(new_marginal)

    @test source_shape(marginal) == source_shape(new_marginal)
    @test action_shape(marginal) == action_shape(new_marginal)
    @test num_target(marginal) == num_target(new_marginal)
    @test state_values(mdp) == state_values(new_mdp)
    @test action_values(mdp) == action_values(new_mdp)

    @test as.lower ≈ new_as.lower
    @test as.gap ≈ new_as.gap

    @test tstates == new_tstates
end
