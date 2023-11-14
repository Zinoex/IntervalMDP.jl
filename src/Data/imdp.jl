
function read_imdp_jl_file(path)
    mdp_or_mc = Dataset(path) do
        n = dataset.attrib["num_states"]
        initial_state = dataset.attrib["initial_state"]
        model = dataset.attrib["model"]

        @assert model ∈ ["imdp", "imc"]
        @assert dataset.attrib["rows"] == "to"
        @assert dataset.attrib["cols"] ∈ ["from", "from/action"]
        @assert dataset.properties["format"] == "sparse_csc"

        lower_colptr = convert.(Int32, dataset["lower_colptr"][:])
        lower_rowval = convert.(Int32, dataset["lower_rowval"][:])
        lower_nzval = dataset["lower_nzval"][:]
        P̲ = SparseMatrixCSC(
            Int32(n + 1),
            Int32(n),
            lower_colptr,
            lower_rowval,
            lower_nzval,
        )

        upper_colptr = convert.(Int32, dataset["upper_colptr"][:])
        upper_rowval = convert.(Int32, dataset["upper_rowval"][:])
        upper_nzval = dataset["upper_nzval"][:]
        P̅ = SparseMatrixCSC(
            Int32(n + 1),
            Int32(n),
            upper_colptr,
            upper_rowval,
            upper_nzval,
        )

        prob = MatrixIntervalProbabilities(; lower = P̲, upper = P̅)

        if model == "imdp"
            return read_imdp_jl_mdp(dataset, prob, initial_state)
        elseif model == "imc"
            return read_imdp_jl_mc(dataset, prob, initial_state)
        end
    end

    return mdp_or_mc
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
