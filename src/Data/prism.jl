

function write_prism_file(path_without_file_ending, mdp, terminal_states)
    write_prism_states_file(path_without_file_ending, mdp)
    write_prism_labels_file(path_without_file_ending, mdp, terminal_states)
    write_prism_transitions_file(path_without_file_ending, mdp)
    write_prism_props_file(path_without_file_ending)
end

function write_prism_states_file(path_without_file_ending, mdp)
    number_states = num_states(mdp)

    lines = Vector{String}(undef, 1 + number_states)
    lines[1] = "(s)"

    for i in 1:number_states
        state = i - 1
        lines[i + 1] = "$state:($state)"
    end

    write(path_without_file_ending * ".sta", join(lines, "\n"))
end

function write_prism_labels_file(path_without_file_ending, mdp, terminal_states)
    istate = initial_state(mdp) - 1

    lines = Vector{String}(undef, 2 + length(terminal_states))
    lines[1] = "0=\"init\" 1=\"deadlock\" 2=\"goal\""
    lines[2] = "$istate: 0"

    for (i, tstate) in enumerate(terminal_states)
        state = tstate - 1  # PRISM uses 0-based indexing
        lines[i + 2] = "$state: 2"
    end

    write(path_without_file_ending * ".lab", join(lines, "\n"))
end

function write_prism_transitions_file(path_without_file_ending, mdp)
    number_states = num_states(mdp)

    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)

    transition_lines = num_src(prob)
    num_transitions = nnz(l)

    sptr = IMDP.stateptr(mdp)
    act = actions(mdp)
    num_choices = length(act)

    open(path_without_file_ending * ".tra", "w") do io
        println(io, "$number_states $num_choices $num_transitions")

        s = 1
        action_idx = 0
        for j in 1:transition_lines
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

function write_prism_props_file(path_without_file_ending)
    line = "Pmaxmin=? [ F \"goal\" ]"

    write(path_without_file_ending * ".pctl", line)
end