using Revise, Test
using IntervalMDP, IntervalMDP.Data, SparseArrays

# Read MDP
mdp, tstates = read_bmdp_tool_file("data/multiObj_robotIMDP.txt")

# Write it back
new_path = tempname() * ".txt"
write_bmdp_tool_file(new_path, mdp, tstates)

# Check the file is there
@test isfile(new_path)

# Read new file and check that the models are the same
new_mdp, new_tstates = read_bmdp_tool_file(new_path)
rm(new_path)

@test num_states(mdp) == num_states(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

@test tstates == new_tstates

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

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

@test tstates == new_tstates
