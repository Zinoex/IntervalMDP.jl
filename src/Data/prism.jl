
function write_prism_file(path_without_file_ending, mdp_or_mc, terminal_states)
    write_prism_states_file(path_without_file_ending, mdp_or_mc)
    write_prism_labels_file(path_without_file_ending, mdp_or_mc, terminal_states)
    write_prism_transitions_file(path_without_file_ending, mdp_or_mc)
    return write_prism_props_file(path_without_file_ending)
end

function write_prism_states_file(path_without_file_ending, mdp_or_mc)
    number_states = num_states(mdp_or_mc)

    lines = Vector{String}(undef, 1 + number_states)
    lines[1] = "(s)"

    for i in 1:number_states
        state = i - 1
        lines[i + 1] = "$state:($state)"
    end

    return write(path_without_file_ending * ".sta", join(lines, "\n"))
end

function write_prism_labels_file(path_without_file_ending, mdp_or_mc, terminal_states)
    istate = initial_state(mdp_or_mc) - 1

    lines = Vector{String}(undef, 2 + length(terminal_states))
    lines[1] = "0=\"init\" 1=\"deadlock\" 2=\"goal\""
    lines[2] = "$istate: 0"

    for (i, tstate) in enumerate(terminal_states)
        state = tstate - 1  # PRISM uses 0-based indexing
        lines[i + 2] = "$state: 2"
    end

    return write(path_without_file_ending * ".lab", join(lines, "\n"))
end

function write_prism_transitions_file(path_without_file_ending, mdp::IntervalMarkovDecisionProcess)
    number_states = num_states(mdp)

    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)

    num_columns = num_src(prob)
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

    num_columns = num_src(prob)
    num_transitions = nnz(l)

    open(path_without_file_ending * ".tra", "w") do io
        println(io, "$number_states 1 $num_transitions")  # number_states number_choices number_transitions

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

function write_prism_props_file(path_without_file_ending)
    line = "Pmaxmin=? [ F \"goal\" ]"

    return write(path_without_file_ending * ".pctl", line)
end
