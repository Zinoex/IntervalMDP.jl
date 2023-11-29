
"""
    write_prism_file(path_without_file_ending, problem)

Write the files required by PRISM explicit engine/format to 
- `path_without_file_ending.sta` (states),
- `path_without_file_ending.lab` (labels),
- `path_without_file_ending.tra` (transitions), and
- `path_without_file_ending.pctl` (properties).

If the specification is a reward optimization problem, then a state rewards file .srew is also written.

See [Data storage formats](@ref) for more information on the file format.
"""
write_prism_file(path_without_file_ending, problem; maximize = true) = 
    write_prism_file(path_without_file_ending, system(problem), specification(problem), satisfaction_mode(problem); maximize)

function write_prism_file(path_without_file_ending, mdp_or_mc, spec, satisfaction_mode; maximize = true)
    write_prism_states_file(path_without_file_ending, mdp_or_mc)
    write_prism_transitions_file(path_without_file_ending, mdp_or_mc)
    write_prism_spec(path_without_file_ending, mdp_or_mc, spec, satisfaction_mode, maximize)

    return nothing
end

function write_prism_states_file(path_without_file_ending, mdp_or_mc)
    number_states = num_states(mdp_or_mc)

    lines = Vector{String}(undef, 1 + number_states)
    lines[1] = "(s)"

    for i in 1:number_states
        state = i - 1
        lines[i + 1] = "$state:($state)"
    end

    write(path_without_file_ending * ".sta", join(lines, "\n"))
end

function write_prism_transitions_file(
    path_without_file_ending,
    mdp::IntervalMarkovDecisionProcess,
)
    number_states = num_states(mdp)

    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)

    num_columns = num_source(prob)
    num_transitions = nnz(l)

    sptr = IMDP.stateptr(mdp)
    act = actions(mdp)
    num_choices = length(act)

    open(path_without_file_ending * ".tra", "w") do io
        println(io, "$number_states $num_choices $num_transitions")

        s = 1
        action_idx = 0
        for j in 1:num_columns
            action = act[j]

            if sptr[s + 1] == j
                s += 1
                action_idx = 0
            end
            src = s - 1

            column_lower = view(l, :, j)
            I, V = SparseArrays.findnz(column_lower)

            for (i, v) in zip(I, V)
                dest = i - 1
                pl = v
                pu = pl + g[i, j]
                pl = max(pl, 1e-12)

                println(io, "$src $action_idx $dest [$pl,$pu] $action")
            end

            action_idx += 1
        end
    end
end

function write_prism_transitions_file(path_without_file_ending, mc::IntervalMarkovChain)
    number_states = num_states(mc)

    prob = transition_prob(mc)
    l, g = lower(prob), gap(prob)

    num_columns = num_source(prob)
    num_transitions = nnz(l)

    open(path_without_file_ending * ".tra", "w") do io
        println(io, "$number_states $number_states $num_transitions")  # number_states number_choices number_transitions

        for j in 1:num_columns
            src = j - 1

            column_lower = view(l, :, j)
            I, V = SparseArrays.findnz(column_lower)

            for (i, v) in zip(I, V)
                dest = i - 1
                pl = v
                pu = pl + g[i, j]
                pl = max(pl, 1e-12)

                println(io, "$src 0 $dest [$pl,$pu] mc")
            end
        end
    end
end

function write_prism_spec(path_without_file_ending, mdp_or_mc, spec, satisfaction_mode, maximize)
    write_prism_labels_file(path_without_file_ending, mdp_or_mc, spec)
    write_prism_rewards_file(path_without_file_ending, spec)
    write_prism_props_file(path_without_file_ending, spec, satisfaction_mode, maximize)
end

function write_prism_labels_file(path_without_file_ending, mdp_or_mc, spec::AbstractReachability)
    istate = initial_state(mdp_or_mc) - 1
    target_states = reach(spec)

    open(path_without_file_ending * ".lab", "w") do io
        println(io, "0=\"init\" 1=\"deadlock\" 2=\"reach\"")
        println(io, "$istate: 0")

        for tstate in target_states
            state = tstate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 2")
        end
    end
end

function write_prism_labels_file(path_without_file_ending, mdp_or_mc, spec::AbstractReachAvoid)
    istate = initial_state(mdp_or_mc) - 1
    target_states = reach(spec)
    avoid_states = avoid(spec)

    open(path_without_file_ending * ".lab", "w") do io
        println(io, "0=\"init\" 1=\"deadlock\" 2=\"reach\" 3=\"avoid\"")
        println(io, "$istate: 0")

        for tstate in target_states
            state = tstate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 2")
        end

        for astate in avoid_states
            state = astate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 3")
        end
    end
end

function write_prism_rewards_file(path_without_file_ending, spec::AbstractReachability)
    # Do nothing - no rewards for reachability
end

function write_prism_rewards_file(path_without_file_ending, spec::AbstractReward)
    # TODO: Implement
end

function write_prism_props_file(path_without_file_ending, spec::FiniteTimeReachability, satisfaction_mode, maximize)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ F<=$(time_horizon(spec)) \"reach\" ]"

    write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(path_without_file_ending, spec::InfiniteTimeReachability, satisfaction_mode, maximize)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ F \"reach\" ]"

    write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(path_without_file_ending, spec::FiniteTimeReachAvoid, satisfaction_mode, maximize)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U<=$(time_horizon(spec)) \"reach\" ]"

    write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(path_without_file_ending, spec::InfiniteTimeReachAvoid, satisfaction_mode, maximize)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U \"reach\" ]"

    write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(path_without_file_ending, spec::FiniteTimeReward, satisfaction_mode, maximize)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U \"reach\" ]"

    write(path_without_file_ending * ".pctl", line)
end

# TODO: Read PRISM
