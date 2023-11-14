
function read_imdp_jl_file(path)
    mdp_or_mc, terminal_states = Dataset(path) do dataset
        n = Int32(dataset.attrib["num_states"] + 1)
        initial_state = Int32(1) # dataset.attrib["initial_state"]
        model = dataset.attrib["model"]

        @assert model ∈ ["imdp", "imc"]
        @assert dataset.attrib["rows"] == "to"
        @assert dataset.attrib["cols"] ∈ ["from", "from/action"]
        @assert dataset.attrib["format"] == "sparse_csc"

        lower_colptr = convert.(Int32, dataset["lower_colptr"][:])
        lower_rowval = convert.(Int32, dataset["lower_rowval"][:])
        lower_nzval = dataset["lower_nzval"][:]
        P̲ = SparseMatrixCSC(
            n,
            n,
            lower_colptr,
            lower_rowval,
            lower_nzval,
        )

        upper_colptr = convert.(Int32, dataset["upper_colptr"][:])
        upper_rowval = convert.(Int32, dataset["upper_rowval"][:])
        upper_nzval = dataset["upper_nzval"][:]
        P̅ = SparseMatrixCSC(
            n,
            n,
            upper_colptr,
            upper_rowval,
            upper_nzval,
        )

        prob = MatrixIntervalProbabilities(; lower = P̲, upper = P̅)
        terminal_states = Int32[n] # convert.(Int32, dataset["terminal_states"][:])

        if model == "imdp"
            return read_imdp_jl_mdp(dataset, prob, initial_state), terminal_states
        elseif model == "imc"
            return read_imdp_jl_mc(dataset, prob, initial_state), terminal_states
        end
    end

    return mdp_or_mc, terminal_states
end

function read_imdp_jl_mdp(dataset, prob, initial_state)
    @assert dataset.attrib["cols"] == "from/action"

    stateptr = convert.(Int32, dataset["stateptr"][:])
    action_vals = dataset["action_vals"][:]

    mdp = IntervalMarkovDecisionProcess(prob, stateptr, action_vals, Int32(initial_state))
    return mdp
end

function read_imdp_jl_mc(dataset, prob, initial_state)
    @assert dataset.attrib["cols"] == "from"

    mc = IntervalMarkovChain(prob, Int32(initial_state))
    return mc
end

function write_imdp_jl_file(path, mdp_or_mc)
    # TODO: implement
end
