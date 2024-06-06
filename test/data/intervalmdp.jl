# Read MDP
mdp = read_intervalmdp_jl_model("data/multiObj_robotIMDP.nc")

# Write it back
new_path = tempname() * ".nc"
write_intervalmdp_jl_model(new_path, mdp)

# Read new file and check that the models are the same
new_mdp = read_intervalmdp_jl_model(new_path)
rm(new_path)

@test num_states(mdp) == num_states(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

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

new_path = tempname() * ".nc"
write_intervalmdp_jl_model(new_path, mc)

# Check the file is there
@test isfile(new_path)

# Read new file and check that the models represent the same system
new_mc = read_intervalmdp_jl_model(new_path)
rm(new_path)

@test new_mc isa IntervalMarkovChain
@test num_states(mc) == num_states(new_mc)

transition_probabilities = transition_prob(mc)
new_transition_probabilities = transition_prob(new_mc)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

# Test specification
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

# Test FiniteTimeReachability
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Minimize)

new_path = tempname() * ".json"
write_intervalmdp_jl_spec(new_path, spec)
@test isfile(new_path)
new_spec = read_intervalmdp_jl_spec(new_path)
rm(new_path)

new_prop = system_property(new_spec)
@test new_prop isa FiniteTimeReachability
@test reach(new_prop) == [3]
@test time_horizon(new_prop) == 10

# Test InfiniteTimeReachability
prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Minimize)

new_path = tempname() * ".json"
write_intervalmdp_jl_spec(new_path, spec)
@test isfile(new_path)
new_spec = read_intervalmdp_jl_spec(new_path)
rm(new_path)

new_prop = system_property(new_spec)
@test new_prop isa InfiniteTimeReachability
@test reach(new_prop) == [3]
@test convergence_eps(new_prop) ≈ 1e-6

# Test FiniteTimeReachAvoid
prop = FiniteTimeReachAvoid([3], [2], 10)
spec = Specification(prop, Pessimistic, Minimize)

new_path = tempname() * ".json"
write_intervalmdp_jl_spec(new_path, spec)
@test isfile(new_path)
new_spec = read_intervalmdp_jl_spec(new_path)
rm(new_path)

new_prop = system_property(new_spec)
@test new_prop isa FiniteTimeReachAvoid
@test reach(new_prop) == [3]
@test avoid(new_prop) == [2]
@test time_horizon(new_prop) == 10

# Test InfiniteTimeReachAvoid
prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
spec = Specification(prop, Pessimistic, Minimize)

new_path = tempname() * ".json"
write_intervalmdp_jl_spec(new_path, spec)
@test isfile(new_path)
new_spec = read_intervalmdp_jl_spec(new_path)
rm(new_path)

new_prop = system_property(new_spec)
@test new_prop isa InfiniteTimeReachAvoid
@test reach(new_prop) == [3]
@test avoid(new_prop) == [2]
@test convergence_eps(new_prop) ≈ 1e-6

# Test FiniteTimeReward
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

# Test InfiniteTimeReward
prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
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
@test convergence_eps(new_prop) ≈ 1e-6
