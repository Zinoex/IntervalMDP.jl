function read_intline(line)
    return parse(Int32, line)
end

function read_bmdp_tool_transition_line(line)
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

"""
    read_bmdp_tool_file(path)

Read a bmdp-tool transition probability file and return an `IntervalMarkovDecisionProcess` and a list of terminal states.
From the file format, it is not clear if the desired reachability verification if the reachability specification is finite
or infinite horizon, the satisfaction_mode is pessimistic or optimistic, or if the actions should minimize or maximize
the probability of reachability.

See [Data storage formats](@ref) for more information on the file format.
"""
function read_bmdp_tool_file(path)
    if splitext(path)[2] != ".txt"
        throw(ArgumentError("A bmdp-tool file must have a .txt extension"))
    end

    open(path, "r") do io
        number_states = read_intline(readline(io))
        number_actions = read_intline(readline(io))
        number_terminal = read_intline(readline(io))

        terminal_states = map(1:number_terminal) do _
            return CartesianIndex(read_intline(readline(io)) + Int32(1))
        end

        probs = Vector{
            IntervalProbabilities{
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

        if isnothing(next)
            throw(ArgumentError("Transitions file is empty"))
        end

        cur_line, state = next
        src, act, dest, lower, upper = read_bmdp_tool_transition_line(cur_line)

        for j in 0:(number_states - 1)
            probs_lower = spzeros(Float64, Int32, number_states, number_actions)
            probs_upper = spzeros(Float64, Int32, number_states, number_actions)

            actions_to_remove = Int64[]

            for k in 0:(number_actions - 1)
                if src != j || act != k
                    push!(actions_to_remove, k + 1)
                end

                while src == j && act == k
                    probs_lower[dest + 1, k + 1] = lower
                    probs_upper[dest + 1, k + 1] = upper

                    next = iterate(lines_it, state)
                    if isnothing(next)
                        break
                    end

                    cur_line, state = next
                    src, act, dest, lower, upper = read_bmdp_tool_transition_line(cur_line)
                end
            end

            actions_to_keep = setdiff(collect(1:number_actions), actions_to_remove)
            probs_lower = probs_lower[:, actions_to_keep]
            probs_upper = probs_upper[:, actions_to_keep]

            probs[j + 1] = IntervalProbabilities(; lower = probs_lower, upper = probs_upper)
        end

        action_list_per_state = collect(0:(number_actions - 1))
        action_list =
            convert.(Int32, mapreduce(_ -> action_list_per_state, vcat, 1:number_states))

        mdp = IntervalMarkovDecisionProcess(probs, action_list)
        return mdp, terminal_states
    end
end

"""
    write_bmdp_tool_file(path, problem::IntervalMDP.AbstractIntervalMDPProblem)

Write a bmdp-tool transition probability file for the given an IMDP and a reachability specification.
The file will not contain enough information to specify a reachability specification. The remaining
parameters are rather command line arguments.

See [Data storage formats](@ref) for more information on the file format.
"""
write_bmdp_tool_file(path, problem::IntervalMDP.AbstractIntervalMDPProblem) =
    write_bmdp_tool_file(path, system(problem), specification(problem))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, spec::Specification)
"""
write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, spec::Specification) =
    write_bmdp_tool_file(path, mdp, system_property(spec))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, prop::AbstractReachability)
"""
write_bmdp_tool_file(
    path,
    mdp::IntervalMarkovProcess,
    prop::IntervalMDP.AbstractReachability,
) = write_bmdp_tool_file(path, mdp, reach(prop))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, terminal_states::Vector{T})
"""
write_bmdp_tool_file(
    path,
    mdp::IntervalMarkovProcess,
    terminal_states::Vector{T},
) where {T} = write_bmdp_tool_file(path, mdp, CartesianIndex.(terminal_states))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovDecisionProcess, terminal_states::Vector{<:CartesianIndex})
"""
function write_bmdp_tool_file(
    path,
    mdp::IntervalMarkovDecisionProcess,
    terminal_states::Vector{<:CartesianIndex},
)
    prob = transition_prob(mdp)
    l, g = lower(prob), gap(prob)
    num_columns = num_source(prob)
    sptr = IntervalMDP.stateptr(mdp)

    number_states = num_states(mdp)
    number_actions = IntervalMDP.max_actions(mdp)
    number_terminal = length(terminal_states)

    open(path, "w") do io
        println(io, number_states)
        println(io, number_actions)
        println(io, number_terminal)

        for terminal_state in terminal_states
            println(io, terminal_state[1] - 1)
        end

        s = 1
        action = 0
        for j in 1:num_columns
            if sptr[s + 1] == j
                s += 1
                action = 0
            end
            src = s - 1

            column_lower = @view l[:, j]
            I, V = SparseArrays.findnz(column_lower)

            for (i, v) in zip(I, V)
                dest = i - 1
                pl = v
                pu = pl + g[i, j]

                transition = "$src $action $dest $pl $pu"
                println(io, transition)
            end

            action += 1
        end
    end
end
