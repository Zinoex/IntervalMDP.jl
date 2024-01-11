mdp, tstates = read_bmdp_tool_file("data/multiObj_robotIMDP.txt")

new_path = tempname() * ".txt"
write_bmdp_tool_file(new_path, mdp, tstates)

@test isfile(new_path)

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

## TODO: test writing an IntervalMarkovChain
## TODO: test writing Problem