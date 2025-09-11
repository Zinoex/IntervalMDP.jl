using Revise, Test
using IntervalMDP, IntervalMDP.Data, SparseArrays

# Read MDP
mdp, tstates = read_bmdp_tool_file("data/multiObj_robotIMDP.txt")

marginal = marginals(mdp)[1]
as = ambiguity_sets(marginal)

@testset "write/read model,tstates" begin
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
    @test state_variables(mdp) == state_variables(new_mdp)
    @test action_variables(mdp) == action_variables(new_mdp)

    @test as.lower ≈ new_as.lower
    @test as.gap ≈ new_as.gap

    @test tstates == new_tstates
end

@testset "write/read problem" begin
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
    @test state_variables(mdp) == state_variables(new_mdp)
    @test action_variables(mdp) == action_variables(new_mdp)

    @test as.lower ≈ new_as.lower
    @test as.gap ≈ new_as.gap

    @test tstates == new_tstates
end