problem = read_prism_file("data/multiObj_robotIMDP")

spec = specification(problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [207]

new_path = tempname()
write_prism_file(new_path, problem)

@test isfile(new_path * ".sta")
@test isfile(new_path * ".tra")
@test isfile(new_path * ".lab")
@test isfile(new_path * ".pctl")

new_problem = read_prism_file(new_path)

mdp, new_mdp = system(problem), system(new_problem)

@test num_states(mdp) == num_states(new_mdp)
@test num_choices(mdp) == num_choices(new_mdp)
@test actions(mdp) == actions(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

spec = specification(new_problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [207]

