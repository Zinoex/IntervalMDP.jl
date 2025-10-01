
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

write_prism_file(path_without_file_ending, mdp, spec) = write_prism_file(
    path_without_file_ending * ".sta",
    path_without_file_ending * ".tra",
    path_without_file_ending * ".lab",
    path_without_file_ending * ".srew",
    path_without_file_ending * ".pctl",
    mdp,
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
    mdp::IntervalMDP.FactoredRMDP,
    spec,
)
    write_prism_states_file(sta_path, mdp)
    write_prism_transitions_file(tra_path, mdp)
    write_prism_spec(lab_path, srew_path, pctl_path, mdp, spec)
end

write_prism_states_file(sta_path, mdp::IntervalMDP.FactoredRMDP) =
    _write_prism_states_file(sta_path, mdp, IntervalMDP.modeltype(mdp))
function _write_prism_states_file(
    sta_path,
    mdp::IntervalMDP.FactoredRMDP,
    ::IntervalMDP.NonFactored,
)
    number_states = num_states(mdp)

    open(sta_path, "w") do io
        println(io, "(s)")

        for i in 1:number_states
            state = i - 1
            println(io, "$state:($i)")
        end
    end
end

write_prism_transitions_file(
    tra_path,
    mdp::IntervalMDP.FactoredRMDP;
    lb_threshold = 1e-12,
) = _write_prism_transitions_file(
    tra_path,
    mdp,
    IntervalMDP.modeltype(mdp);
    lb_threshold = lb_threshold,
)

function _write_prism_transitions_file(
    tra_path,
    mdp::IntervalMDP.FactoredRMDP,
    ::IntervalMDP.IsIMDP;
    lb_threshold,
)
    marginal = marginals(mdp)[1]

    num_transitions = nnz(ambiguity_sets(marginal).lower)  # Number of non-zero entries in the lower bound matrix
    num_choices = source_shape(marginal)[1] * action_shape(marginal)[1]

    open(tra_path, "w") do io
        println(io, "$(num_states(mdp)) $num_choices $num_transitions")

        for jₛ in CartesianIndices(source_shape(marginal))
            src = jₛ[1] - 1  # PRISM uses 0-based indexing

            for jₐ in CartesianIndices(action_shape(marginal))
                act = jₐ[1] - 1  # PRISM uses 0-based indexing
                ambiguity_set = marginal[jₐ, jₛ]

                for i in support(ambiguity_set)
                    dest = i - 1  # PRISM uses 0-based indexing
                    pl = max(lower(ambiguity_set, i), lb_threshold)  # PRISM requires constant support
                    pu = upper(ambiguity_set, i)

                    println(io, "$src $act $dest [$pl,$pu]")
                end
            end
        end
    end
end

function write_prism_spec(lab_path, srew_path, pctl_path, mdp, spec)
    write_prism_labels_file(lab_path, mdp, system_property(spec))
    write_prism_rewards_file(srew_path, mdp, system_property(spec))
    write_prism_props_file(pctl_path, spec)
end

function write_prism_labels_file(lab_path, mdp, prop::IntervalMDP.AbstractReachability)
    istates = initial_states(mdp)
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

function write_prism_labels_file(lab_path, mdp, prop::IntervalMDP.AbstractReachAvoid)
    istates = initial_states(mdp)
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

function write_prism_labels_file(lab_path, mdp, prop::IntervalMDP.AbstractReward)
    istates = initial_states(mdp)

    open(lab_path, "w") do io
        println(io, "0=\"init\" 1=\"deadlock\"")

        for istate in istates
            state = istate - 1  # PRISM uses 0-based indexing
            println(io, "$state: 0")
        end
    end
end

function write_prism_rewards_file(lab_path, mdp, prop::IntervalMDP.AbstractReachability)
    # Do nothing - no rewards for reachability
    return nothing
end

function write_prism_rewards_file(srew_path, mdp, prop::IntervalMDP.AbstractReward)
    rew = reward(prop)

    open(srew_path, "w") do io
        println(io, "$(num_states(mdp)) $(length(rew))")

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
    probs, num_actions = read_prism_transitions_file(tra_path, num_states)
    initial_states, spec = read_prism_spec(lab_path, srew_path, pctl_path, num_states)

    mdp = IntervalMarkovDecisionProcess(probs, num_actions, initial_states)

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

        if num_states != num_states_t
            throw(
                DimensionMismatch(
                    "Number of states in .sta file ($num_states) does not match number of states in .tra file ($num_states_t).",
                ),
            )
        end

        if num_choices <= 0
            throw(ArgumentError("Number of choices must be positive, was $num_choices."))
        end

        if num_transitions <= 0
            throw(
                ArgumentError(
                    "Number of transitions must be positive, was $num_transitions.",
                ),
            )
        end

        if num_choices % num_states_t != 0
            throw(
                ArgumentError(
                    "Number of choices ($num_choices) must be a multiple of the number of states ($num_states_t).",
                ),
            )
        end
        num_actions = num_choices ÷ num_states_t
        num_src_states = num_choices ÷ num_actions

        probs_lower = Vector{SparseVector{Float64, Int32}}(undef, num_choices)
        probs_upper = Vector{SparseVector{Float64, Int32}}(undef, num_choices)

        lines_it = eachline(io)

        next = iterate(lines_it)
        if isnothing(next)
            throw(ArgumentError("Transitions file is empty"))
        end

        cur_line, state = next
        src, act, dest, lower, upper, _ = read_prism_transition_line(cur_line)

        for jₛ in 1:num_src_states
            for jₐ in 1:num_actions
                state_action_probs_lower = spzeros(Float64, Int32, num_states)
                state_action_probs_upper = spzeros(Float64, Int32, num_states)

                if src != jₛ - 1
                    throw(
                        ArgumentError(
                            "Transitions file is not sorted by source index or the number of actions was less than expected. Expected source index $(jₛ - 1), got $src.",
                        ),
                    )
                end

                if act != jₐ - 1
                    throw(
                        ArgumentError(
                            "Transitions file is not sorted by action index or the number of actions was less than expected. Expected action index $(jₐ - 1), got $act.",
                        ),
                    )
                end

                while src == jₛ - 1 && act == jₐ - 1
                    # PRISM uses 0-based indexing
                    state_action_probs_lower[dest + 1] = lower
                    state_action_probs_upper[dest + 1] = upper

                    next = iterate(lines_it, state)
                    if isnothing(next)
                        break
                    end

                    cur_line, state = next
                    src, act, dest, lower, upper, _ = read_prism_transition_line(cur_line)
                end

                j = (jₛ - 1) * num_actions + jₐ
                probs_lower[j] = state_action_probs_lower
                probs_upper[j] = state_action_probs_upper
            end
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

        probs = IntervalAmbiguitySets(; lower = probs_lower, upper = probs_upper)

        return probs, num_actions
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
