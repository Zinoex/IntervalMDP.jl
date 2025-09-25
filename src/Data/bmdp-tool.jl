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
        num_states = read_intline(readline(io))
        num_actions = read_intline(readline(io))
        num_terminal = read_intline(readline(io))

        terminal_states = map(1:num_terminal) do _
            return CartesianIndex(read_intline(readline(io)) + Int32(1))
        end

        num_choices = num_states * num_actions

        probs_lower = Vector{SparseVector{Float64, Int32}}(undef, num_choices)
        probs_upper = Vector{SparseVector{Float64, Int32}}(undef, num_choices)

        lines_it = eachline(io)
        next = iterate(lines_it)

        if isnothing(next)
            throw(ArgumentError("Transitions file is empty"))
        end

        cur_line, state = next
        src, act, dest, lower, upper = read_bmdp_tool_transition_line(cur_line)

        for jₛ in 1:num_states
            for jₐ in 1:num_actions
                state_action_probs_lower = spzeros(Float64, Int32, num_states)
                state_action_probs_upper = spzeros(Float64, Int32, num_states)

                if src != jₛ - 1
                    throw(ArgumentError("Transitions file is not sorted by source index or the number of actions was less than expected. Expected source index $(jₛ - 1), got $src."))
                end

                if act != jₐ - 1
                    throw(ArgumentError("Transitions file is not sorted by action index or the number of actions was less than expected. Expected action index $(jₐ - 1), got $act."))
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
                    src, act, dest, lower, upper = read_bmdp_tool_transition_line(cur_line)
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

        mdp = IntervalMarkovDecisionProcess(probs, num_actions)
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
write_bmdp_tool_file(path, mdp::IntervalMDP.IntervalMarkovProcess, spec::Specification) =
    write_bmdp_tool_file(path, mdp, system_property(spec))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, prop::AbstractReachability)
"""
write_bmdp_tool_file(
    path,
    mdp::IntervalMDP.IntervalMarkovProcess,
    prop::IntervalMDP.AbstractReachability,
) = write_bmdp_tool_file(path, mdp, reach(prop))

"""
    write_bmdp_tool_file(path, mdp::IntervalMarkovProcess, terminal_states::Vector{T})
"""
write_bmdp_tool_file(
    path,
    mdp::IntervalMDP.IntervalMarkovProcess,
    terminal_states::Vector{T},
) where {T} = write_bmdp_tool_file(path, mdp, CartesianIndex.(terminal_states))

"""
    write_bmdp_tool_file(path, mdp::IMDP, terminal_states::Vector{<:CartesianIndex})
"""
write_bmdp_tool_file(
    path,
    mdp::IntervalMDP.FactoredRMDP,
    terminal_states::Vector{<:CartesianIndex},
) = _write_bmdp_tool_file(path, mdp, IntervalMDP.modeltype(mdp), terminal_states)

function _write_bmdp_tool_file(
    path,
    mdp::IntervalMDP.FactoredRMDP,
    ::IntervalMDP.IsIMDP,
    terminal_states::Vector{<:CartesianIndex},
)
    marginal = marginals(mdp)[1]

    number_states = num_states(mdp)
    number_actions = IntervalMDP.num_actions(mdp)
    number_terminal = length(terminal_states)

    open(path, "w") do io
        println(io, number_states)
        println(io, number_actions)
        println(io, number_terminal)

        for terminal_state in terminal_states
            println(io, terminal_state[1] - 1)
        end

        for jₛ in CartesianIndices(source_shape(marginal))
            src = jₛ[1] - 1
            for jₐ in CartesianIndices(action_shape(marginal))
                act = jₐ[1] - 1
                ambiguity_set = marginal[jₐ, jₛ]

                for i in support(ambiguity_set)
                    dest = i - 1  # bmdp-tool uses 0-based indexing
                    pl = lower(ambiguity_set, i)
                    pu = upper(ambiguity_set, i)

                    println(io, "$src $act $dest $pl $pu")
                end
            end
        end
    end
end
