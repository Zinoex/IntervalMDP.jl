function read_intline(line)
    return parse(Int32, line)
end

function read_transition_line(line)
    words = eachsplit(line; limit = 5)

    (item, state) = iterate(words)
    src = parse(Int32, item)

    (item, state) = iterate(words, state)
    act = parse(Int32, item)

    (item, state) = iterate(words, state)
    dest = parse(Int32, item)

    (item, state) = iterate(words, state)
    lower = parse(Float64, item)

    (item, state) = iterate(words, state)
    upper = parse(Float64, item)

    return src, act, dest, lower, upper
end

function read_bmdp_tool_file(path)
    open(path, "r") do io
        number_states = read_intline(readline(io))
        number_actions = read_intline(readline(io))
        number_terminal = read_intline(readline(io))

        terminal_states = map(1:number_terminal) do _
            return read_intline(readline(io)) + Int32(1)
        end

        probs = Vector{
            MatrixIntervalProbabilities{
                Float64,
                Vector{Float64},
                SparseArrays.FixedSparseCSC{Float64, Int32},
            },
        }(
            undef,
            number_states,
        )

        lines_it = eachline(io)
        next = iterate(lines_it)

        if !isnothing(next)
            cur_line, state = next
            src, act, dest, lower, upper = read_transition_line(cur_line)
        end

        for j in 0:(number_states - 1)
            probs_lower = spzeros(Float64, Int32, number_states, number_actions)
            probs_upper = spzeros(Float64, Int32, number_states, number_actions)

            for k in 0:(number_actions - 1)
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
            end

            probs[j + 1] = IntervalProbabilities(; lower = probs_lower, upper = probs_upper)
        end

        action_list_per_state = collect(0:(number_actions - 1))
        action_list =
            convert.(Int32, mapreduce(_ -> action_list_per_state, vcat, 1:number_states))

        mdp = IntervalMarkovDecisionProcess(probs, action_list, Int32(1))
        return mdp, terminal_states
    end
end

function write_bmdp_tool_file(path, mdp::IntervalMarkovDecisionProcess, terminal_states)
    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)
    num_columns = num_src(prob)
    sptr = IMDP.stateptr(mdp)
    act = actions(mdp)

    number_states = num_states(mdp)
    number_actions = length(unique(act))
    number_terminal = length(terminal_states)

    open(path, "w") do io
        println(io, number_states)
        println(io, number_actions)
        println(io, number_terminal)

        for terminal_state in terminal_states
            println(io, terminal_state - 1)
        end

        s = 1
        for j in 1:num_columns
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
                println(io, transition)
            end
        end
    end
end

function write_bmdp_tool_file(path, mdp::IntervalMarkovChain, terminal_states)
    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)
    num_columns = num_src(prob)

    number_states = num_states(mdp)
    number_terminal = length(terminal_states)

    open(path, "w") do io
        println(io, number_states)
        println(io, 1)  # number_actions
        println(io, number_terminal)

        for terminal_state in terminal_states
            println(io, terminal_state - 1)
        end

        for j in 1:num_columns
            src = j - 1

            column_lower = view(l, :, j)
            I, V = SparseArrays.findnz(column_lower)

            for (i, v) in zip(I, V)
                dest = i - 1
                pl = v
                pu = pl + g[i, j]

                transition = "$src 0 $dest $pl $pu"
                println(io, transition)
            end
        end
    end
end
