
function read_imdp_jl_file(path)
    mdp_or_mc, terminal_states = Dataset(path) do dataset
        n = Int32(dataset.attrib["num_states"])
        initial_state = dataset.attrib["initial_state"]
        model = dataset.attrib["model"]

        @assert model ∈ ["imdp", "imc"]
        @assert dataset.attrib["rows"] == "to"
        @assert dataset.attrib["cols"] ∈ ["from", "from/action"]
        @assert dataset.attrib["format"] == "sparse_csc"

        lower_colptr = convert.(Int32, dataset["lower_colptr"][:])
        lower_rowval = convert.(Int32, dataset["lower_rowval"][:])
        lower_nzval = dataset["lower_nzval"][:]
        P̲ = SparseMatrixCSC(n, n, lower_colptr, lower_rowval, lower_nzval)

        upper_colptr = convert.(Int32, dataset["upper_colptr"][:])
        upper_rowval = convert.(Int32, dataset["upper_rowval"][:])
        upper_nzval = dataset["upper_nzval"][:]
        P̅ = SparseMatrixCSC(n, n, upper_colptr, upper_rowval, upper_nzval)

        prob = IntervalProbabilities(; lower = P̲, upper = P̅)
        terminal_states = convert.(Int32, dataset["terminal_states"][:])

        if model == "imdp"
            return read_imdp_jl_mdp(dataset, prob, initial_state), terminal_states
        elseif model == "imc"
            return read_imdp_jl_mc(dataset, prob, initial_state), terminal_states
        end
    end

    return mdp_or_mc, terminal_states
end

function read_imdp_jl_mdp(dataset, prob, initial_state)
    @assert dataset.attrib["model"] == "imdp"
    @assert dataset.attrib["cols"] == "from/action"

    stateptr = convert.(Int32, dataset["stateptr"][:])
    action_vals = dataset["action_vals"][:]

    mdp = IntervalMarkovDecisionProcess(prob, stateptr, action_vals, Int32(initial_state))
    return mdp
end

function read_imdp_jl_mc(dataset, prob, initial_state)
    @assert dataset.attrib["model"] == "imc"
    @assert dataset.attrib["cols"] == "from"

    mc = IntervalMarkovChain(prob, Int32(initial_state))
    return mc
end

function write_imdp_jl_file(path, mdp_or_mc, terminal_states)
    Dataset(path, "c") do dataset
        dataset.attrib["format"] = "sparse_csc"
        dataset.attrib["num_states"] = num_states(mdp_or_mc)
        dataset.attrib["rows"] = "to"
        dataset.attrib["initial_state"] = initial_state(mdp_or_mc)

        prob = transition_prob(mdp_or_mc)
        l = lower(prob)
        g = gap(prob)

        defDim(dataset, "lower_colptr", length(l.colptr))
        v = defVar(dataset, "lower_colptr", Int32, ("lower_colptr",))
        v[:] = l.colptr

        defDim(dataset, "lower_rowval", length(l.rowval))
        v = defVar(dataset, "lower_rowval", Int32, ("lower_rowval",))
        v[:] = l.rowval

        defDim(dataset, "lower_nzval", length(l.nzval))
        v = defVar(dataset, "lower_nzval", eltype(l.nzval), ("lower_nzval",))
        v[:] = l.nzval

        defDim(dataset, "upper_colptr", length(g.colptr))
        v = defVar(dataset, "upper_colptr", Int32, ("upper_colptr",))
        v[:] = g.colptr

        defDim(dataset, "upper_rowval", length(g.rowval))
        v = defVar(dataset, "upper_rowval", Int32, ("upper_rowval",))
        v[:] = g.rowval

        defDim(dataset, "upper_nzval", length(g.nzval))
        v = defVar(dataset, "upper_nzval", eltype(g.nzval), ("upper_nzval",))
        v[:] = l.nzval + g.nzval

        defDim(dataset, "terminal_states", length(terminal_states))
        v = defVar(dataset, "terminal_states", Int32, ("terminal_states",))
        v[:] = terminal_states

        return write_imdp_jl_model_specific(dataset, mdp_or_mc)
    end
end

function write_imdp_jl_model_specific(dataset, mdp::IntervalMarkovDecisionProcess)
    dataset.attrib["model"] = "imdp"
    dataset.attrib["cols"] = "from/action"

    defDim(dataset, "stateptr", length(stateptr(mdp)))
    v = defVar(dataset, "stateptr", Int32, ("stateptr",))
    v[:] = stateptr(mdp)

    defDim(dataset, "action_vals", length(actions(mdp)))
    v = defVar(dataset, "action_vals", eltype(actions(mdp)), ("action_vals",))
    return v[:] = actions(mdp)
end

function write_imdp_jl_model_specific(dataset, mc::IntervalMarkovChain)
    dataset.attrib["model"] = "imc"
    return dataset.attrib["cols"] = "from"
end
