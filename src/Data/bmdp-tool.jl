
function readnumbers(types, line)
    words = split(line)

    @assert length(words) == length(types)

    return [parse(t, w) for (t, w) in zip(types, words)]
end

function read_bmdp_tool_file(path)
    lines = readlines(path)

    number_states = readnumbers([Int32], lines[1])[1]
    number_actions = readnumbers([Int32], lines[2])[1]
    number_terminal = readnumbers([Int32], lines[3])[1]

    terminal_states = first.(readnumbers.(tuple([Int32]), lines[4:4 + number_terminal - 1])) .+ Int32(1)

    probs = Vector{MatrixIntervalProbabilities{Float64}}(undef, number_states)

    cur_line = 4 + number_terminal
    for j in 0:number_states - 1
        probs_lower = spzeros(Float64, Int32, number_states, number_actions) 
        probs_upper = spzeros(Float64, Int32, number_states, number_actions)

        for k in 0:number_actions - 1
            src, act, dest, lower, upper = readnumbers([Int32, Int32, Int32, Float64, Float64], lines[cur_line])

            while src == j && act == k
                probs_lower[dest + 1, k + 1] = lower
                probs_upper[dest + 1, k + 1] = upper

                cur_line += 1
                if cur_line > length(lines)
                    break
                end

                src, act, dest, lower, upper = readnumbers([Int32, Int32, Int32, Float64, Float64], lines[cur_line])
            end
        end

        probs[j + 1] = MatrixIntervalProbabilities(;lower=probs_lower, upper=probs_upper)
    end

    action_list_per_state = 0:number_actions - 1
    action_list = convert.(Int32, mapreduce(_ -> action_list_per_state, vcat, 1:number_states))

    mdp = IntervalMarkovDecisionProcess(probs, action_list, Int32(1))
    return mdp, terminal_states
end


function write_bmdp_tool_file(path, mdp::IntervalMarkovDecisionProcess, terminal_states)
    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)
    nsrc = num_src(prob)
    sptr = IMDP.stateptr(mdp)
    act = actions(mdp)

    number_states = num_states(mdp)
    number_actions = length(unique(act))
    number_terminal = length(terminal_states)

    lines = Vector{String}(undef, 3 + number_terminal + nnz(g))

    lines[1] = string(number_states)
    lines[2] = string(number_actions)
    lines[3] = string(number_terminal)

    lines[4:4 + number_terminal - 1] = string.(terminal_states .- 1)


    s = 1
    cur_line = 4 + number_terminal
    for j in 1:nsrc
        action = act[j]

        if sptr[s + 1] == j
            s += 1
        end
        src = s - 1

        column_lower = view(l, :, j)
        I, V = SparseArrays.findnz(column_lower)

        for (i, v) in zip(I, V)
            dest = i - 1
            pl = v
            pu = pl + g[i, j]

            transition = "$src $action $dest $pl $pu"

            lines[cur_line] = transition
            cur_line += 1
        end
    end

    write(path, join(lines, "\n"))
end