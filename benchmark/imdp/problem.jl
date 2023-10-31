using Revise, BenchmarkTools, ProgressMeter
using Random, StatsBase
using IMDP, SparseArrays, CUDA, Adapt

function readnumbers(types, line)
    words = split(line)

    @assert length(words) == length(types)

    return [parse(t, w) for (t, w) in zip(types, words)]
end

function create_problem()
    lines = readlines(joinpath(@__DIR__, "multiObj_robotIMDP.txt"))

    number_states = readnumbers([Int32], lines[1])[1]
    number_actions = readnumbers([Int32], lines[2])[1]
    number_terminal = readnumbers([Int32], lines[3])[1]

    terminal_states = first.(readnumbers.(tuple([Int32]), lines[4:4 + number_terminal - 1]))

    probs = Vector{MatrixIntervalProbabilities{Float64}}(undef, number_states)

    cur_line = 4 + number_terminal
    for j in 0:number_states - 1
        probs_lower = spzeros(number_states, number_actions)
        probs_upper = spzeros(number_states, number_actions)

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

    action_list_per_state = collect(0:number_actions - 1)
    action_list = mapreduce(_ -> action_list_per_state, vcat, 1:number_states)

    mdp = IntervalMarkovDecisionProcess(probs, action_list, 0)
    problem = Problem(mdp, InfiniteTimeReachability(terminal_states, number_states, 1e-6))

    return problem
end

prob = create_problem()