
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
write_prism_file(path_without_file_ending, problem; maximize = true) = write_prism_file(
    path_without_file_ending,
    system(problem),
    specification(problem),
    satisfaction_mode(problem);
    maximize,
)

function write_prism_file(
    path_without_file_ending,
    mdp_or_mc,
    spec,
    satisfaction_mode;
    maximize = true,
)
    write_prism_states_file(path_without_file_ending, mdp_or_mc)
    write_prism_transitions_file(path_without_file_ending, mdp_or_mc)
    write_prism_spec(path_without_file_ending, mdp_or_mc, spec, satisfaction_mode, maximize)

    return nothing
end

function write_prism_states_file(path_without_file_ending, mdp_or_mc)
    number_states = num_states(mdp_or_mc)

    open(path_without_file_ending * ".sta", "w") do io
        println(io, "$number_states")

        for i in 1:number_states
            state = i - 1
            println(io, "$state")
        end
    end
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

function write_prism_spec(
    path_without_file_ending,
    mdp_or_mc,
    spec,
    satisfaction_mode,
    maximize,
)
    write_prism_labels_file(path_without_file_ending, mdp_or_mc, spec)
    write_prism_rewards_file(path_without_file_ending, mdp_or_mc, spec)
    return write_prism_props_file(
        path_without_file_ending,
        spec,
        satisfaction_mode,
        maximize,
    )
end

function write_prism_labels_file(
    path_without_file_ending,
    mdp_or_mc,
    spec::AbstractReachability,
)
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

function write_prism_labels_file(
    path_without_file_ending,
    mdp_or_mc,
    spec::AbstractReachAvoid,
)
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

function write_prism_rewards_file(
    path_without_file_ending,
    mdp_or_mc,
    spec::AbstractReachability,
)
    # Do nothing - no rewards for reachability
end

function write_prism_rewards_file(path_without_file_ending, mdp_or_mc, spec::AbstractReward)
    rew = reward(spec)

    open(path_without_file_ending * ".srew", "w") do io
        println(io, "$(num_states(mdp_or_mc)) $(length(rew))")

        for (i, r) in enumerate(rew)
            s = i - 1  # PRISM uses 0-based indexing
            println(io, "$s $r")
        end
    end
end

function write_prism_props_file(
    path_without_file_ending,
    spec::FiniteTimeReachability,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ F<=$(time_horizon(spec)) \"reach\" ]"

    return write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(
    path_without_file_ending,
    spec::InfiniteTimeReachability,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ F \"reach\" ]"

    return write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(
    path_without_file_ending,
    spec::FiniteTimeReachAvoid,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U<=$(time_horizon(spec)) \"reach\" ]"

    return write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(
    path_without_file_ending,
    spec::InfiniteTimeReachAvoid,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U \"reach\" ]"

    return write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(
    path_without_file_ending,
    spec::FiniteTimeReward,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "R$strategy$adversary=? [ C<=$(time_horizon(spec)) ]"

    return write(path_without_file_ending * ".pctl", line)
end

function write_prism_props_file(
    path_without_file_ending,
    spec::InfiniteTimeReward,
    satisfaction_mode,
    maximize,
)
    strategy = maximize ? "max" : "min"
    adversary = (satisfaction_mode == Optimistic) ? "max" : "min"

    line = "R$strategy$adversary=? [ C ]"

    return write(path_without_file_ending * ".pctl", line)
end

"""
    read_prism_file(path_without_file_ending)

Read PRISM explicit file formats and pctl file, and return a Problem including system and specification.

See [PRISM Explicit Model Files](https://prismmodelchecker.org/manual/Appendices/ExplicitModelFiles) for more information on the file format.
"""
function read_prism_file(path_without_file_ending)
    num_states = read_prism_states_file(path_without_file_ending)
    prob = read_prism_transitions_file(path_without_file_ending, num_states)
    initial_state, spec = read_prism_spec(path_without_file_ending, num_states)

    mdp = IntervalMarkovDecisionProcess(prob, initial_state)

    return Problem(mdp, spec)
end

function read_prism_states_file(path_without_file_ending)
    num_states = open(path_without_file_ending * ".sta", "r") do io
        return countlines(io) - 1
    end

    return num_states
end

function read_prism_transitions_file(path_without_file_ending, num_states)
    open(path_without_file_ending * ".tra", "r") do io
        num_states_t, num_choices, num_transitions =
            read_prism_transitions_file_header(readline(io))

        @assert num_states == num_states_t

        probs = Vector{
            MatrixIntervalProbabilities{
                Float64,
                Vector{Float64},
                SparseArrays.FixedSparseCSC{Float64, Int32},
            },
        }(
            undef,
            num_states,
        )
        actions = Vector{Any}(undef, num_choices)

        lines_it = eachline(io)
        next = iterate(lines_it)

        if !isnothing(next)
            cur_line, state = next
            src, act_index, dest, lower, upper, act = read_prism_transition_line(cur_line)
        end

        for j in 0:(num_states - 1)
            probs_lower = spzeros(Float64, Int32, num_states, number_actions)
            probs_upper = spzeros(Float64, Int32, num_states, number_actions)

            if isnothing(next)
                break
            end

            while src == j && act == k
                probs_lower[dest + 1, k + 1] = lower
                probs_upper[dest + 1, k + 1] = upper

                next = iterate(lines_it, state)
                if isnothing(next)
                    break
                end

                cur_line, state = next
                src, act, dest, lower, upper = read_transition_line(cur_line)
            end

            probs[j + 1] = IntervalProbabilities(; lower = probs_lower, upper = probs_upper)
        end

        action_list_per_state = collect(0:(number_actions - 1))
        return action_list =
            convert.(Int32, mapreduce(_ -> action_list_per_state, vcat, 1:number_states))
    end
end

function read_prism_transitions_file_header(line)
    words = eachsplit(line; limit = 3)

    (item, state) = iterate(words)
    num_states = parse(Int32, item)

    (item, state) = iterate(words, state)
    num_choices = parse(Int32, item)

    (item, state) = iterate(words, state)
    num_transitions = parse(Int32, item)

    return num_states, num_choices, num_transitions
end

function read_prism_transition_line(line)
    words = eachsplit(line; limit = 5)

    (item, state) = iterate(words)
    src = parse(Int32, item)

    (item, state) = iterate(words, state)
    act_idx = parse(Int32, item)

    (item, state) = iterate(words, state)
    dest = parse(Int32, item)

    (item, state) = iterate(words, state)
    lower_upper = eachsplit(item[2:(end - 1)], ","; limit = 2)

    (item, lu_state) = iterate(lower_upper)
    lower = parse(Float64, item)

    (item, lu_state) = iterate(lower_upper, lu_state)
    upper = parse(Float64, item)

    word = iterate(words, state)
    if isnothing(word)
        act = act_idx
    else
        (item, state) = word
        act = item
    end

    return src, act_index, dest, lower, upper, act
end

function read_prism_spec(path_without_file_ending, num_states)
    prop_type, prop_meta, satisfaction_mode, strategy_mode =
        read_prism_props_file(path_without_file_ending)

    # Possibly read the rewards file, if the property is a reward optimization problem.
    rewards = read_prism_rewards_file(path_without_file_ending, prop_type, num_states)

    # Read at least the initial state but possibly also the reach and avoid sets.
    prop, initial_state =
        read_prism_labels_file(path_without_file_ending, prop_type, prop_meta, rewards)
    spec = Specification(prop, satisfaction_mode, strategy_mode)

    return initial_state, spec
end

function read_prism_props_file(path_without_file_ending)
    property = read(path_without_file_ending * ".pctl")
    m = match(
        r"(?<probrew>(P|R))(?<strategy>max|min)(?<adversary>max|min)=\? \[ (?<pathprop>.+) \]",
        property,
    )
    strategy_mode = m[:strategy] == "max" ? Maximize : Minimize
    satisfaction_mode = (m[:adversary] == "min") ? Pessimistic : Optimistic

    if m[:probrew] == "P"
        abstract_type = AbstractReachability
    elseif m[:probrew] == "R"
        abstract_Type = AbstractReward
    else
        throw(ValueError("Incorrect property $property"))
    end

    prop_type, prop_meta = read_prism_path_prob(abstract_type, m[:pathprop])

    return prop_type, prop_meta, satisfaction_mode, strategy_mode
end

function read_prism_path_prob(::Type{AbstractReachability}, pathprop)
    eps = 1e-6

    m = match(r"F \"reach\"", pathprop)
    if !isnothing(m)
        return InfiniteTimeReachability, eps
    end

    m = match(r"F<=(?<time_horizon>\d+) \"reach\"", pathprop)
    if !isnothing(m)
        return FiniteTimeReachability, parse_intline(m[:time_horizon])
    end

    m = match(r"!\"avoid\" U \"reach\"", pathprop)
    if !isnothing(m)
        return InfiniteTimeReachAvoid, eps
    end

    m = match(r"!\"avoid\" U<=(?<time_horizon>\d+) \"reach\"", pathprop)
    if !isnothing(m)
        return FiniteTimeReachAvoid, m[:time_horizon]
    end

    throw(ValueError("Invalid path property $pathprop"))
end

function read_prism_path_prob(::Type{AbstractReward}, pathprop)
    m = match(r"C", pathprop)
    if !isnothing(m)
        eps = 1e-6
        return InfiniteTimeReward, eps
    end

    m = match(r"C<=(?<time_horizon>\d+)", pathprop)
    if !isnothing(m)
        return FiniteTimeReward, read_intline(m[:time_horizon])
    end

    throw(ValueError("Invalid path property $pathprop"))
end

read_prism_rewards_file(
    path_without_file_ending,
    prop_type::Type{AbstractReachability},
    num_states,
) = nothing

function read_prism_rewards_header(line)
    words = eachsplit(line; limit = 2)

    (item, state) = iterate(words)
    num_rewards = parse(Int32, item)

    (item, state) = iterate(words, state)
    num_nonzero_rewards = parse(Int32, item)

    return num_rewards, num_nonzero_rewards
end

function read_prism_rewards_line(line)
    words = eachsplit(line; limit = 2)

    (item, state) = iterate(words)
    index = parse(Int32, item)

    (item, state) = iterate(words, state)
    reward = parse(Float64, item)

    return index, reward
end

function read_prism_rewards_file(
    path_without_file_ending,
    prop_type::Type{AbstractReward},
    num_states,
)
    return open(path_without_file_ending * ".pctl", "r") do io
        num_rewards, num_nonzero_rewards = read_prism_rewards_header(readline(io))

        @assert num_rewards == num_states "The number of rewards must match the number of states"

        rewards = zeros(Float64, num_rewards)
        for _ in 1:num_nonzero_rewards
            index, reward = read_prism_rewards_line(readline(io))
            rewards[index] = reward
        end

        return rewards
    end
end

function read_prism_labels_file(
    path_without_file_ending,
    spec_type::Type{AbstractReachability},
    spec_meta,
    rewards,
)
    state_labels = read_prism_labels(path_without_file_ending)
    initial_state = find_initial_state(state_labels)

    reach = map(first, findall((k, v) -> "reach" in V))
    spec = spec_type(reach, spec_meta...)

    return spec, initial_state
end

function read_prism_labels_file(
    path_without_file_ending,
    spec_type::Type{AbstractReachAvoid},
    spec_meta,
    rewards,
)
    state_labels = read_prism_labels(path_without_file_ending)
    initial_state = find_initial_state(state_labels)

    reach = map(first, findall((k, v) -> "reach" in V))
    avoid = map(first, findall((k, v) -> "avoid" in V))
    spec = spec_type(reach, avoid, spec_meta...)

    return spec, initial_state
end

function read_prism_labels_file(
    path_without_file_ending,
    spec_type::Type{AbstractReward},
    spec_meta,
    rewards,
)
    state_labels = read_prism_labels(path_without_file_ending)
    initial_state = find_initial_state(state_labels)

    spec = spec_type(rewards, spec_meta...)

    return spec, initial_state
end

function read_prism_labels(path_without_file_ending)
    return open(path_without_file_ending * ".lab", "r") do io
        labels = read_prism_labels_header(readline(io))

        state_labels = Dict{Int32, Vector{String}}()

        for line in eachline(io)
            words = split(line)

            state_index = parse(Int32, words[1][1:(end - 1)])
            state_labels = map(words[2:end]) do word
                label_index = parse(Int32, word)
                return labels[label_index]
            end

            state_indices[state_index] = state_labels
        end

        return state_labels
    end
end

function read_prism_labels_header(line)
    words = eachsplit(line)

    words = Dict(map(words) do word
        terms = eachsplit(word; limit = 2)

        (item, state) = iterate(terms)
        index = parse(Int32, item)

        (item, state) = iterate(terms)
        label = item[2:(end - 1)]

        return index => label
    end)

    return words
end

function find_initial_state(state_labels)
    initial_states = map(first, findall((k, v) -> "init" in V))

    @assert length(initial_states) == 1

    return initial_states[1]
end
