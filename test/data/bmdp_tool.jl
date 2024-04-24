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
@test num_choices(mdp) == num_choices(new_mdp)
@test actions(mdp) == actions(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

@test tstates == new_tstates

# Write IntervalMarkovChain
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(3, [2, 3], [0.1, 0.2]),
        SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
        SparseVector(3, [3], [1.0]),
    ),
    upper = sparse_hcat(
        SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
        SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
        SparseVector(3, [3], [1.0]),
    ),
)

mc = IntervalMarkovChain(prob, [1])
tstates = [3]

new_path = tempname() * ".txt"
write_bmdp_tool_file(new_path, mc, tstates)

# Check the file is there
@test isfile(new_path)

# Read new file and check that the models represent the same system
new_mdp, new_tstates = read_bmdp_tool_file(new_path)
rm(new_path)

@test num_states(mc) == num_states(new_mdp)
@test num_states(mc) == num_choices(new_mdp)
@test actions(new_mdp) == [0, 0, 0]

transition_probabilities = transition_prob(mc)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

@test tstates == new_tstates

# Write problem
tstates = [207]
prop = FiniteTimeReachability(tstates, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)

new_path = tempname() * ".txt"
write_bmdp_tool_file(new_path, problem)

# Check the file is there
@test isfile(new_path)

# Read new file and check that the models represent the same system
new_mdp, new_tstates = read_bmdp_tool_file(new_path)

@test num_states(mdp) == num_states(new_mdp)
@test num_choices(mdp) == num_choices(new_mdp)
@test actions(mdp) == actions(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

@test tstates == new_tstates
