
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
write_prism_file(path_without_file_ending, problem) =
    write_prism_file(path_without_file_ending, system(problem), specification(problem))

write_prism_file(path_without_file_ending, mdp_or_mc, spec) = write_prism_file(
    path_without_file_ending * ".sta",
    path_without_file_ending * ".tra",
    path_without_file_ending * ".lab",
    path_without_file_ending * ".srew",
    path_without_file_ending * ".pctl",
    mdp_or_mc,
    spec,
)

write_prism_file(sta_path, tra_path, lab_path, pctl_path, problem) =
    write_prism_file(sta_path, tra_path, lab_path, missing, pctl_path, problem)

write_prism_file(sta_path, tra_path, lab_path, srew_path, pctl_path, problem) =
    write_prism_file(
        sta_path,
        tra_path,
        lab_path,
        srew_path,
        pctl_path,
        system(problem),
        specification(problem),
    )

function write_prism_file(
    sta_path,
    tra_path,
    lab_path,
    srew_path,
    pctl_path,
    mdp_or_mc,
    spec,
)
    write_prism_states_file(sta_path, mdp_or_mc)
    write_prism_transitions_file(tra_path, mdp_or_mc)
    write_prism_spec(lab_path, srew_path, pctl_path, mdp_or_mc, spec)
end

function write_prism_states_file(sta_path, mdp_or_mc)
    number_states = num_states(mdp_or_mc)

    open(sta_path, "w") do io
        println(io, "(s)")

        for i in 1:number_states
            state = i - 1
            println(io, "$state:($i)")
        end
    end
end

function write_prism_transitions_file(tra_path, mdp::IntervalMarkovDecisionProcess)
    number_states = num_states(mdp)

    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)

    num_columns = num_source(prob)
    num_transitions = nnz(l)

    sptr = IntervalMDP.stateptr(mdp)
    num_choices = num_columns

    open(tra_path, "w") do io
        println(io, "$number_states $num_choices $num_transitions")

        s = 1
        action_idx = 0
        for j in 1:num_columns
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

                println(io, "$src $action_idx $dest [$pl,$pu] $j")
            end

            action_idx += 1
        end
    end
end

function write_prism_spec(lab_path, srew_path, pctl_path, mdp_or_mc, spec)
    write_prism_labels_file(lab_path, mdp_or_mc, system_property(spec))
    write_prism_rewards_file(srew_path, mdp_or_mc, system_property(spec))
    write_prism_props_file(pctl_path, spec)
end

function write_prism_labels_file(
    lab_path,
    mdp_or_mc,
    prop::IntervalMDP.AbstractReachability,
)
    istates = initial_states(mdp_or_mc)
    target_states = reach(prop)

    open(lab_path, "w") do io
        println(io, "0=\"init\" 1=\"deadlock\" 2=\"reach\"")

        for istate in istates
            state = istate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 0")
        end

        for tstate in target_states
            state = tstate[1] - 1  # PRISM uses 0-based indexing
            println(io, "$state: 2")
        end
    end
end

function write_prism_labels_file(lab_path, mdp_or_mc, prop::IntervalMDP.AbstractReachAvoid)
    istates = initial_states(mdp_or_mc)
    target_states = reach(prop)
    avoid_states = avoid(prop)

    open(lab_path, "w") do io
        println(io, "0=\"init\" 1=\"deadlock\" 2=\"reach\" 3=\"avoid\"")

        for istate in istates
            state = istate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 0")
        end

        for tstate in target_states
            state = tstate[1] - 1  # PRISM uses 0-based indexing
            println(io, "$state: 2")
        end

        for astate in avoid_states
            state = astate[1] - 1  # PRISM uses 0-based indexing
            println(io, "$state: 3")
        end
    end
end

function write_prism_labels_file(lab_path, mdp_or_mc, prop::IntervalMDP.AbstractReward)
    istates = initial_states(mdp_or_mc)

    open(lab_path, "w") do io
        println(io, "0=\"init\" 1=\"deadlock\"")

        for istate in istates
            state = istate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 0")
        end
    end
end

function write_prism_rewards_file(
    lab_path,
    mdp_or_mc,
    prop::IntervalMDP.AbstractReachability,
)
    # Do nothing - no rewards for reachability
    return nothing
end

function write_prism_rewards_file(srew_path, mdp_or_mc, prop::IntervalMDP.AbstractReward)
    rew = reward(prop)

    open(srew_path, "w") do io
        println(io, "$(num_states(mdp_or_mc)) $(length(rew))")

        for (i, r) in enumerate(rew)
            s = i - 1  # PRISM uses 0-based indexing
            println(io, "$s $r")
        end
    end
end

function write_prism_props_file(pctl_path, spec::Specification{<:FiniteTimeReachability})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    prop = system_property(spec)
    line = "P$strategy$adversary=? [ F<=$(time_horizon(prop)) \"reach\" ]"

    return write(pctl_path, line)
end

function write_prism_props_file(pctl_path, spec::Specification{<:InfiniteTimeReachability})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ F \"reach\" ]"

    return write(pctl_path, line)
end

function write_prism_props_file(pctl_path, spec::Specification{<:FiniteTimeReachAvoid})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    prop = system_property(spec)
    line = "P$strategy$adversary=? [ !\"avoid\" U<=$(time_horizon(prop)) \"reach\" ]"

    return write(pctl_path, line)
end

function write_prism_props_file(pctl_path, spec::Specification{<:InfiniteTimeReachAvoid})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    line = "P$strategy$adversary=? [ !\"avoid\" U \"reach\" ]"

    return write(pctl_path, line)
end

function write_prism_props_file(pctl_path, spec::Specification{<:FiniteTimeReward})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    prop = system_property(spec)
    line = "R$strategy$adversary=? [ C<=$(time_horizon(prop)) ]"

    return write(pctl_path, line)
end

function write_prism_props_file(pctl_path, spec::Specification{<:InfiniteTimeReward})
    strategy = (strategy_mode(spec) == Maximize) ? "max" : "min"
    adversary = (satisfaction_mode(spec) == Optimistic) ? "max" : "min"

    line = "R$strategy$adversary=? [ C ]"

    return write(pctl_path, line)
end

"""
    read_prism_file(path_without_file_ending)

Read PRISM explicit file formats and pctl file, and return a ControlSynthesisProblem including system and specification.

See [PRISM Explicit Model Files](https://prismmodelchecker.org/manual/Appendices/ExplicitModelFiles) for more information on the file format.
"""
read_prism_file(path_without_file_ending) = read_prism_file(
    path_without_file_ending * ".sta",
    path_without_file_ending * ".tra",
    path_without_file_ending * ".lab",
    path_without_file_ending * ".srew",
    path_without_file_ending * ".pctl",
)

read_prism_file(sta_path, tra_path, lab_path, pctl_path) =
    read_prism_file(sta_path, tra_path, lab_path, missing, pctl_path)

function read_prism_file(sta_path, tra_path, lab_path, srew_path, pctl_path)
    num_states = read_prism_states_file(sta_path)
    probs, stateptr = read_prism_transitions_file(tra_path, num_states)
    initial_states, spec = read_prism_spec(lab_path, srew_path, pctl_path, num_states)

    mdp = IntervalMarkovDecisionProcess(probs, stateptr, initial_states)

    return ControlSynthesisProblem(mdp, spec)
end

function read_prism_states_file(sta_path)
    num_states = open(sta_path, "r") do io
        return countlines(io) - 1
    end

    return num_states
end

function read_prism_transitions_file(tra_path, num_states)
    open(tra_path, "r") do io
        num_states_t, num_choices, num_transitions =
            read_prism_transitions_file_header(readline(io))

        @assert num_states == num_states_t

        probs_lower = Vector{SparseVector{Float64, Int32}}(undef, num_choices)
        probs_upper = Vector{SparseVector{Float64, Int32}}(undef, num_choices)

        stateptr = Vector{Int32}(undef, num_states + 1)
        stateptr[1] = 1
        stateptr[end] = num_choices + 1

        lines_it = eachline(io)
        next = iterate(lines_it)

        if isnothing(next)
            throw(ArgumentError("Transitions file is empty"))
        end

        cur_line, state = next
        # We ignore the act field since we only use indices for actions/choices
        src, act_idx, dest, lower, upper, act = read_prism_transition_line(cur_line)

        outer_src = src

        for j in 1:num_choices
            state_action_probs_lower = spzeros(Float64, Int32, num_states)
            state_action_probs_upper = spzeros(Float64, Int32, num_states)

            cur_src = src
            cur_act_idx = act_idx

            if src != outer_src
                # PRISM uses 0-based indexing
                stateptr[src + 1] = j
                outer_src = src
            end

            while src == cur_src && act_idx == cur_act_idx
                # PRISM uses 0-based indexing
                state_action_probs_lower[dest + 1] = lower
                state_action_probs_upper[dest + 1] = upper

                next = iterate(lines_it, state)
                if isnothing(next)
                    break
                end

                cur_line, state = next
                src, act_idx, dest, lower, upper, act = read_prism_transition_line(cur_line)
            end

            probs_lower[j] = state_action_probs_lower
            probs_upper[j] = state_action_probs_upper
        end

        # Colptr is the same for both lower and upper
        num_col = mapreduce(x -> size(x, 2), +, probs_lower)
        colptr = zeros(Int32, num_col + 1)
        nnz_sofar = 0
        @inbounds for i in eachindex(probs_lower)
            colptr[i] = nnz_sofar + 1
            nnz_sofar += nnz(probs_lower[i])
        end
        colptr[end] = nnz_sofar + 1

        probs_lower_rowval = mapreduce(lower -> lower.nzind, vcat, probs_lower)
        probs_lower_nzval = mapreduce(lower -> lower.nzval, vcat, probs_lower)
        probs_lower = SparseMatrixCSC(
            num_states,
            num_col,
            colptr,
            probs_lower_rowval,
            probs_lower_nzval,
        )

        probs_upper_rowval = mapreduce(upper -> upper.nzind, vcat, probs_upper)
        probs_upper_nzval = mapreduce(upper -> upper.nzval, vcat, probs_upper)
        probs_upper = SparseMatrixCSC(
            num_states,
            num_col,
            colptr,
            probs_upper_rowval,
            probs_upper_nzval,
        )

        probs = IntervalProbabilities(; lower = probs_lower, upper = probs_upper)

        return probs, stateptr
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

    return src, act_idx, dest, lower, upper, act
end

function read_prism_spec(lab_path, srew_path, pctl_path, num_states)
    prop_type, prop_meta, satisfaction_mode, strategy_mode =
        read_prism_props_file(pctl_path)

    # Possibly read the rewards file, if the property is a reward optimization problem.
    rewards = read_prism_rewards_file(srew_path, prop_type, num_states)

    # Read at least the initial state but possibly also the reach and avoid sets.
    prop, initial_state = read_prism_labels_file(lab_path, prop_type, prop_meta, rewards)
    spec = Specification(prop, satisfaction_mode, strategy_mode)

    return initial_state, spec
end

function read_prism_props_file(pctl_path)
    property = read(pctl_path, String)
    m = match(
        r"(?<probrew>(P|R))(?<strategy>max|min)(?<adversary>max|min)=\? \[ (?<pathprop>.+) \]",
        property,
    )
    strategy_mode = m[:strategy] == "max" ? Maximize : Minimize
    satisfaction_mode = (m[:adversary] == "min") ? Pessimistic : Optimistic

    if m[:probrew] == "P"
        abstract_type = IntervalMDP.AbstractReachability
    elseif m[:probrew] == "R"
        abstract_type = IntervalMDP.AbstractReward
    else
        throw(DomainError("Incorrect property $property"))
    end

    prop_type, prop_meta = read_prism_path_prob(abstract_type, m[:pathprop])

    return prop_type, prop_meta, satisfaction_mode, strategy_mode
end

function read_prism_path_prob(::Type{IntervalMDP.AbstractReachability}, pathprop)
    convergence_eps = 1e-6

    m = match(r"!\"avoid\" U<=(?<time_horizon>\d+) \"reach\"", pathprop)
    if !isnothing(m)
        return FiniteTimeReachAvoid, (read_intline(m[:time_horizon]),)
    end

    m = match(r"!\"avoid\" U \"reach\"", pathprop)
    if !isnothing(m)
        return InfiniteTimeReachAvoid, (convergence_eps,)
    end

    m = match(r"F<=(?<time_horizon>\d+) \"reach\"", pathprop)
    if !isnothing(m)
        return FiniteTimeReachability, (read_intline(m[:time_horizon]),)
    end

    m = match(r"F \"reach\"", pathprop)
    if !isnothing(m)
        return InfiniteTimeReachability, (convergence_eps,)
    end

    throw(DomainError("Invalid path property $pathprop"))
end

function read_prism_path_prob(::Type{<:IntervalMDP.AbstractReward}, pathprop)
    m = match(r"C<=(?<time_horizon>\d+)", pathprop)
    if !isnothing(m)
        discount = 1.0  # This does not need to converge (since finite time)
        return FiniteTimeReward, (discount, read_intline(m[:time_horizon]))
    end

    m = match(r"C", pathprop)
    if !isnothing(m)
        discount = 1.0  # This is not guaranteed to converge (it must be in (0, 1) but PRISM does not support that)
        convergence_eps = 1e-6
        return InfiniteTimeReward, (discount, convergence_eps)
    end

    throw(DomainError("Invalid path property $pathprop"))
end

read_prism_rewards_file(
    srew_path,
    prop_type::Type{<:IntervalMDP.AbstractReachability},
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
    srew_path,
    prop_type::Type{<:IntervalMDP.AbstractReward},
    num_states,
)
    return open(srew_path, "r") do io
        num_rewards, num_nonzero_rewards = read_prism_rewards_header(readline(io))

        @assert num_rewards == num_states "The number of rewards must match the number of states"

        rewards = zeros(Float64, num_rewards)
        for _ in 1:num_nonzero_rewards
            index, reward = read_prism_rewards_line(readline(io))
            rewards[index + 1] = reward
        end

        return rewards
    end
end

function read_prism_labels_file(
    lab_path,
    prop_type::Type{<:IntervalMDP.AbstractReachability},
    prop_meta,
    rewards,
)
    state_labels = read_prism_labels(lab_path)
    initial_states = find_initial_states(state_labels)

    reach = find_states_label(state_labels, "reach")
    prop = prop_type(reach, prop_meta...)

    return prop, initial_states
end

function read_prism_labels_file(
    lab_path,
    prop_type::Type{<:IntervalMDP.AbstractReachAvoid},
    prop_meta,
    rewards,
)
    state_labels = read_prism_labels(lab_path)
    initial_states = find_initial_states(state_labels)

    reach = find_states_label(state_labels, "reach")
    avoid = find_states_label(state_labels, "avoid")
    prop = prop_type(reach, avoid, prop_meta...)

    return prop, initial_states
end

function read_prism_labels_file(
    lab_path,
    prop_type::Type{<:IntervalMDP.AbstractReward},
    prop_meta,
    rewards,
)
    state_labels = read_prism_labels(lab_path)
    initial_states = find_initial_states(state_labels)

    prop = prop_type(rewards, prop_meta...)

    return prop, initial_states
end

function read_prism_labels(lab_path)
    return open(lab_path, "r") do io
        global_labels = read_prism_labels_header(readline(io))
        state_labels = Dict{Int32, Vector{String}}()

        for line in eachline(io)
            words = split(line)

            # PRISM uses 0-based indexing
            index = parse(Int32, words[1][1:(end - 1)]) + 1

            labels = map(words[2:end]) do word
                label_index = parse(Int32, word)
                return global_labels[label_index]
            end

            state_labels[index] = labels
        end

        return state_labels
    end
end

function read_prism_labels_header(line)
    words = eachsplit(line)

    words = Dict(map(words) do word
        terms = eachsplit(word, "="; limit = 2)

        (item, state) = iterate(terms)
        index = parse(Int32, item)

        (item, state) = iterate(terms, state)
        label = strip(item, ['"'])

        return index => label
    end)

    return words
end

find_states_label(state_labels, label) =
    collect(keys(filter(((k, v),) -> label in v, state_labels)))
find_initial_states(state_labels) = find_states_label(state_labels, "init")
