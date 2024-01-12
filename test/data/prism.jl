## Read PRISM file
problem = read_prism_file("data/multiObj_robotIMDP")

# Check the specification
spec = specification(problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [207]

# Write back to a new path
new_path = tempname()
write_prism_file(new_path, problem)

# Check files exist
@test isfile(new_path * ".sta")
@test isfile(new_path * ".tra")
@test isfile(new_path * ".lab")
@test isfile(new_path * ".pctl")

# Read back
new_problem = read_prism_file(new_path)
rm(new_path * ".sta")
rm(new_path * ".tra")
rm(new_path * ".lab")
rm(new_path * ".pctl")

# Check the two systems match
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

## Explicit file paths
problem = read_prism_file("data/multiObj_robotIMDP")

# Check the specification
spec = specification(problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [207]

# Write back to a new path
new_path = tempname()
sta_path = new_path * "a.sta"
tra_path = new_path * "b.tra"
lab_path = new_path * "c.lab"
pctl_path = new_path * "d.pctl"
write_prism_file(sta_path, tra_path, lab_path, pctl_path, problem)

# Check files exist
@test isfile(new_path * "a.sta")
@test isfile(new_path * "b.tra")
@test isfile(new_path * "c.lab")
@test isfile(new_path * "d.pctl")

# Read back
new_problem = read_prism_file(sta_path, tra_path, lab_path, pctl_path)
rm(new_path * "a.sta")
rm(new_path * "b.tra")
rm(new_path * "c.lab")
rm(new_path * "d.pctl")

# Check the two systems match
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

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mc, spec)

new_path = tempname()
write_prism_file(new_path, problem)

# Check the file is there
@test isfile(new_path * ".sta")
@test isfile(new_path * ".tra")
@test isfile(new_path * ".lab")
@test isfile(new_path * ".pctl")

# Read new file and check that the models represent the same system
new_problem = read_prism_file(new_path)
rm(new_path * ".sta")
rm(new_path * ".tra")
rm(new_path * ".lab")
rm(new_path * ".pctl")

new_mdp = system(new_problem)

@test num_states(mc) == num_states(new_mdp)
@test num_states(mc) == num_choices(new_mdp)
@test actions(new_mdp) == ["mc", "mc", "mc"]

transition_probabilities = transition_prob(mc)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

spec = specification(new_problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa FiniteTimeReachability
@test reach(prop) == [3]
@test time_horizon(prop) == 10

## Explicit file paths for reward prop
prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mc, spec)

# Write back to a new path
new_path = tempname()
sta_path = new_path * "a.sta"
tra_path = new_path * "b.tra"
lab_path = new_path * "c.lab"
srew_path = new_path * "d.srew"
pctl_path = new_path * "e.pctl"
write_prism_file(sta_path, tra_path, lab_path, srew_path, pctl_path, problem)

# Check files exist
@test isfile(new_path * "a.sta")
@test isfile(new_path * "b.tra")
@test isfile(new_path * "c.lab")
@test isfile(new_path * "d.srew")
@test isfile(new_path * "e.pctl")

# Read back
new_problem = read_prism_file(sta_path, tra_path, lab_path, srew_path, pctl_path)
rm(new_path * "a.sta")
rm(new_path * "b.tra")
rm(new_path * "c.lab")
rm(new_path * "d.srew")
rm(new_path * "e.pctl")

# Check the two systems match
new_mdp = system(new_problem)

@test num_states(mc) == num_states(new_mdp)
@test num_states(mc) == num_choices(new_mdp)

transition_probabilities = transition_prob(mc)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

spec = specification(new_problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa FiniteTimeReward
@test reward(prop) ≈ [1.0, 2.0, 3.0]
@test discount(prop) ≈ 1.0

## TODO: Test reading and writing various properties
