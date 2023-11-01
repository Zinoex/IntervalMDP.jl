

function write_prism_file(path, mdp, terminal_states)
    number_states = num_states(mdp)

    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)
    transition_lines = num_src(prob)
    sptr = IMDP.stateptr(mdp)
    act = actions(mdp)

    lines = Vector{String}(undef, 10 + transition_lines)

    lines[1] = "mdp"
    lines[2] = ""
    lines[3] = "module M"
    lines[4] = ""
    lines[5] = "s:[0..$(number_states - 1)] init $(initial_state(mdp) - 1);"
    lines[6] = ""

    s = 1
    cur_line = 7
    for j in 1:transition_lines
        action = act[j]

        if sptr[s + 1] == j
            s += 1
        end
        src = s - 1

        column_lower = view(l, :, j)
        I, V = SparseArrays.findnz(column_lower)

        to = join(map(zip(I, V)) do (i, v)
            dest = i - 1
            pl = v
            pu = pl + g[i, j]

            return "[$pl, $pu]:(s'=$dest)"
        end, " + ")
        transition = "[$action] s=$src -> $to;"

        lines[cur_line] = transition
        cur_line += 1
    end

    lines[end - 3] = ""
    lines[end - 2] = "endmodule"
    lines[end - 1] = ""
    lines[end] = "label \"goal\" = $(join(["s=$(terminal_state - 1)" for terminal_state in terminal_states], "|"));"

    write(path, join(lines, "\n"))
end