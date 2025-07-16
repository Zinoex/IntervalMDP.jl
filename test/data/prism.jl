using Revise, Test
using IntervalMDP, IntervalMDP.Data, SparseArrays

## Read PRISM file
problem = read_prism_file("data/multiObj_robotIMDP")

# Check the specification
spec = specification(problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [CartesianIndex(207)]

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
@test reach(prop) == [CartesianIndex(207)]

## Explicit file paths
problem = read_prism_file("data/multiObj_robotIMDP")

# Check the specification
spec = specification(problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa InfiniteTimeReachability
@test reach(prop) == [CartesianIndex(207)]

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
@test reach(prop) == [CartesianIndex(207)]

## Explicit file paths for reward prop
prop = FiniteTimeReward(collect(1.0:207.0), 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = VerificationProblem(mdp, spec)

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

@test num_states(mdp) == num_states(new_mdp)

transition_probabilities = transition_prob(mdp)
new_transition_probabilities = transition_prob(new_mdp)

@test size(transition_probabilities) == size(new_transition_probabilities)
@test lower(transition_probabilities) ≈ lower(new_transition_probabilities)
@test gap(transition_probabilities) ≈ gap(new_transition_probabilities)

spec = specification(new_problem)
@test satisfaction_mode(spec) == Pessimistic
@test strategy_mode(spec) == Maximize

prop = system_property(spec)
@test prop isa FiniteTimeReward
@test reward(prop) ≈ collect(1.0:207.0)
@test discount(prop) ≈ 1.0

## Write and read reach avoid labels
new_path = tempname()
prop = FiniteTimeReachAvoid([3], [2], 10)

IntervalMDP.Data.write_prism_labels_file(new_path, mdp, prop)
prop, istates =
    IntervalMDP.Data.read_prism_labels_file(new_path, FiniteTimeReachAvoid, (10,), nothing)

@test prop isa FiniteTimeReachAvoid
@test reach(prop) == [CartesianIndex(3)]
@test avoid(prop) == [CartesianIndex(2)]
@test time_horizon(prop) == 10

## Write and read FTR min/max min/max
prop = FiniteTimeReachability([3], 10)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachability
@test prop_meta == (10,)
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachability
@test prop_meta == (10,)
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachability
@test prop_meta == (10,)
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachability
@test prop_meta == (10,)
@test sat_mode == Optimistic
@test strat_mode == Maximize

## Write and read ITR min/max min/max
prop = InfiniteTimeReachability([3], 1e-6)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachability
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachability
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachability
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachability
@test sat_mode == Optimistic
@test strat_mode == Maximize

## Write and read FTRA min/max min/max
prop = FiniteTimeReachAvoid([3], [2], 10)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachAvoid
@test prop_meta == (10,)
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachAvoid
@test prop_meta == (10,)
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachAvoid
@test prop_meta == (10,)
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReachAvoid
@test prop_meta == (10,)
@test sat_mode == Optimistic
@test strat_mode == Maximize

## Write and read ITRA min/max min/max
prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachAvoid
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachAvoid
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachAvoid
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReachAvoid
@test sat_mode == Optimistic
@test strat_mode == Maximize

## Write and read FTDR min/max min/max
prop = FiniteTimeReward([1.0, 2.0, 3.0], 0.9, 10)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReward
@test prop_meta == (1.0, 10)
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReward
@test prop_meta == (1.0, 10)
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReward
@test prop_meta == (1.0, 10)
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == FiniteTimeReward
@test prop_meta == (1.0, 10)
@test sat_mode == Optimistic
@test strat_mode == Maximize

## Write and read ITDR min/max min/max
prop = InfiniteTimeReward([1.0, 2.0, 3.0], 0.9, 1e-6)
pes_min_spec = Specification(prop, Pessimistic, Minimize)
pes_max_spec = Specification(prop, Pessimistic, Maximize)
opt_min_spec = Specification(prop, Optimistic, Minimize)
opt_max_spec = Specification(prop, Optimistic, Maximize)

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReward
@test sat_mode == Pessimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, pes_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReward
@test sat_mode == Pessimistic
@test strat_mode == Maximize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_min_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReward
@test sat_mode == Optimistic
@test strat_mode == Minimize

new_path = tempname()
IntervalMDP.Data.write_prism_props_file(new_path, opt_max_spec)
@test isfile(new_path)
prop_type, prop_meta, sat_mode, strat_mode =
    IntervalMDP.Data.read_prism_props_file(new_path)

@test prop_type == InfiniteTimeReward
@test sat_mode == Optimistic
@test strat_mode == Maximize
