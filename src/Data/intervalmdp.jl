"""
    read_intervalmdp_jl_file(path)

Read an IntervalMDP.jl data file and return an `IntervalMarkovDecisionProcess`
or `IntervalMarkovChain` and a list of terminal states. 

See [Data storage formats](@ref) for more information on the file format.
"""
function read_intervalmdp_jl(model_path, spec_path)
    mdp_or_mc = read_intervalmdp_jl_model(model_path)
    spec = read_intervalmdp_jl_spec(spec_path)

    return Problem(mdp_or_mc, spec)
end

"""
    read_intervalmdp_jl_model(model_path)

Read an `IntervalMarkovDecisionProcess` or `IntervalMarkovChain` from an IntervalMDP.jl system file (netCDF sparse format).

See [Data storage formats](@ref) for more information on the file format.
"""
function read_intervalmdp_jl_model(model_path)
    mdp_or_mc = Dataset(model_path) do dataset
        n = Int32(dataset.attrib["num_states"])
        model = dataset.attrib["model"]

        @assert model ∈ ["imdp", "imc"]
        @assert dataset.attrib["rows"] == "to"
        @assert dataset.attrib["cols"] ∈ ["from", "from/action"]
        @assert dataset.attrib["format"] == "sparse_csc"

        initial_states = convert.(Int32, dataset["initial_states"][:])

        lower_colptr = convert.(Int32, dataset["lower_colptr"][:])
        lower_rowval = convert.(Int32, dataset["lower_rowval"][:])
        lower_nzval = dataset["lower_nzval"][:]
        P̲ = SparseMatrixCSC(n, length(lower_colptr) - 1, lower_colptr, lower_rowval, lower_nzval)

        upper_colptr = convert.(Int32, dataset["upper_colptr"][:])
        upper_rowval = convert.(Int32, dataset["upper_rowval"][:])
        upper_nzval = dataset["upper_nzval"][:]
        P̅ = SparseMatrixCSC(n, length(upper_colptr) - 1, upper_colptr, upper_rowval, upper_nzval)

        prob = IntervalProbabilities(; lower = P̲, upper = P̅)

        if model == "imdp"
            return read_intervalmdp_jl_mdp(dataset, prob, initial_states)
        elseif model == "imc"
            return read_intervalmdp_jl_mc(dataset, prob, initial_states)
        end
    end

    return mdp_or_mc
end

function read_intervalmdp_jl_mdp(dataset, prob, initial_states)
    @assert dataset.attrib["model"] == "imdp"
    @assert dataset.attrib["cols"] == "from/action"

    stateptr = convert.(Int32, dataset["stateptr"][:])
    action_vals = dataset["action_vals"][:]

    mdp = IntervalMarkovDecisionProcess(prob, stateptr, action_vals, initial_states)
    return mdp
end

function read_intervalmdp_jl_mc(dataset, prob, initial_states)
    @assert dataset.attrib["model"] == "imc"
    @assert dataset.attrib["cols"] == "from"

    mc = IntervalMarkovChain(prob, initial_states)
    return mc
end

"""
    read_intervalmdp_jl_spec(spec_path)

Read a `Specification` from an IntervalMDP.jl spec file (JSON-format).

See [Data storage formats](@ref) for more information on the file format.
"""
function read_intervalmdp_jl_spec(spec_path)
    data = JSON.parsefile(spec_path; inttype = Int32)

    prop = read_intervalmdp_jl_property(data["property"])

    if data["satisfaction_mode"] == "optimistic"
        satisfaction_mode = Optimistic
    elseif data["satisfaction_mode"] == "pessimistic"
        satisfaction_mode = Pessimistic
    else
        throw(
            ValueError(
                "Invalid satisfaction mode: $(data["satisfaction_mode"]). Expected \"optimistic\" or \"pessimistic\".",
            ),
        )
    end

    if data["strategy_mode"] == "minimize"
        strategy_mode = Minimize
    elseif data["strategy_mode"] == "maximize"
        strategy_mode = Maximize
    else
        throw(
            ValueError(
                "Invalid strategy mode: $(data["strategy_mode"]). Expected \"minimize\" or \"maximize\".",
            ),
        )
    end

    return Specification(prop, satisfaction_mode, strategy_mode)
end

function read_intervalmdp_jl_property(prop_dict)
    if prop_dict["type"] == "reachability"
        return read_intervalmdp_jl_reachability_property(prop_dict)
    elseif prop_dict["type"] == "reach-avoid"
        return read_intervalmdp_jl_reach_avoid_property(prop_dict)
    elseif prop_dict["type"] == "reward"
        return read_intervalmdp_jl_reward_property(prop_dict)
    else
        throw(
            ValueError(
                "Invalid property_type: $(data["type"]). Expected \"reachability\", \"reach-avoid\", or \"reward\".",
            ),
        )
    end
end

function read_intervalmdp_jl_reachability_property(prop_dict)
    @assert prop_dict["type"] == "reachability"

    reach = convert(Vector{Int32}, prop_dict["reach"])

    if prop_dict["infinite_time"]
        return InfiniteTimeReachability(reach, prop_dict["eps"])
    else
        return FiniteTimeReachability(reach, prop_dict["time_horizon"])
    end
end

function read_intervalmdp_jl_reach_avoid_property(prop_dict)
    @assert prop_dict["type"] == "reach-avoid"

    reach = convert(Vector{Int32}, prop_dict["reach"])
    avoid = convert(Vector{Int32}, prop_dict["avoid"])

    if prop_dict["infinite_time"]
        return InfiniteTimeReachAvoid(reach, avoid, prop_dict["eps"])
    else
        return FiniteTimeReachAvoid(reach, avoid, prop_dict["time_horizon"])
    end
end

function read_intervalmdp_jl_reward_property(prop_dict)
    @assert prop_dict["type"] == "reward"

    reward = convert(Vector{Float64}, prop_dict["reward"])
    discount = prop_dict["discount"]

    if prop_dict["infinite_time"]
        return InfiniteTimeReward(reward, discount, prop_dict["eps"])
    else
        return FiniteTimeReward(reward, discount, prop_dict["time_horizon"])
    end
end

"""
    write_intervalmdp_jl_model(model_path, mdp_or_mc)

Write an `IntervalMarkovDecisionProcess` or `IntervalMarkovChain` to an IntervalMDP.jl system file (netCDF sparse format).

See [Data storage formats](@ref) for more information on the file format.
"""
function write_intervalmdp_jl_model(model_path, mdp_or_mc::IntervalMarkovProcess)
    Dataset(model_path, "c") do dataset
        dataset.attrib["format"] = "sparse_csc"
        dataset.attrib["num_states"] = num_states(mdp_or_mc)
        dataset.attrib["rows"] = "to"

        istates = initial_states(mdp_or_mc)
        defDim(dataset, "initial_states", length(istates))
        v = defVar(dataset, "initial_states", Int32, ("initial_states",); deflatelevel = 5)
        v[:] = istates

        prob = transition_prob(mdp_or_mc)
        l = lower(prob)
        g = gap(prob)

        defDim(dataset, "lower_colptr", length(l.colptr))
        v = defVar(dataset, "lower_colptr", Int32, ("lower_colptr",); deflatelevel = 5)
        v[:] = l.colptr

        defDim(dataset, "lower_rowval", length(l.rowval))
        v = defVar(dataset, "lower_rowval", Int32, ("lower_rowval",); deflatelevel = 5)
        v[:] = l.rowval

        defDim(dataset, "lower_nzval", length(l.nzval))
        v = defVar(
            dataset,
            "lower_nzval",
            eltype(l.nzval),
            ("lower_nzval",);
            deflatelevel = 5,
        )
        v[:] = l.nzval

        defDim(dataset, "upper_colptr", length(g.colptr))
        v = defVar(dataset, "upper_colptr", Int32, ("upper_colptr",); deflatelevel = 5)
        v[:] = g.colptr

        defDim(dataset, "upper_rowval", length(g.rowval))
        v = defVar(dataset, "upper_rowval", Int32, ("upper_rowval",); deflatelevel = 5)
        v[:] = g.rowval

        defDim(dataset, "upper_nzval", length(g.nzval))
        v = defVar(
            dataset,
            "upper_nzval",
            eltype(g.nzval),
            ("upper_nzval",);
            deflatelevel = 5,
        )
        v[:] = l.nzval + g.nzval

        return write_intervalmdp_jl_model_specific(dataset, mdp_or_mc)
    end
end

write_intervalmdp_jl_model(model_path, problem::Problem) =
    write_intervalmdp_jl_model(model_path, system(problem))

function write_intervalmdp_jl_model_specific(dataset, mdp::IntervalMarkovDecisionProcess)
    dataset.attrib["model"] = "imdp"
    dataset.attrib["cols"] = "from/action"

    defDim(dataset, "stateptr", length(IntervalMDP.stateptr(mdp)))
    v = defVar(dataset, "stateptr", Int32, ("stateptr",))
    v[:] = IntervalMDP.stateptr(mdp)

    defDim(dataset, "action_vals", length(actions(mdp)))
    v = defVar(dataset, "action_vals", eltype(actions(mdp)), ("action_vals",))
    return v[:] = actions(mdp)
end

function write_intervalmdp_jl_model_specific(dataset, mc::IntervalMarkovChain)
    dataset.attrib["model"] = "imc"
    return dataset.attrib["cols"] = "from"
end

"""
    write_intervalmdp_jl_spec(spec_path, spec::Specification)

Write a `Specification` to an IntervalMDP.jl spec file (JSON-format).

See [Data storage formats](@ref) for more information on the file format.
"""
function write_intervalmdp_jl_spec(spec_path, spec::Specification; indent=4)
    data = Dict(
        "property" => intervalmdp_jl_property_dict(system_property(spec)),
        "satisfaction_mode" =>
            intervalmdp_jl_satisfaction_mode(satisfaction_mode(spec)),
        "strategy_mode" => intervalmdp_jl_strategy_mode(strategy_mode(spec)),
    )

    open(spec_path, "w") do io
        return JSON.print(io, data, indent)
    end
end

write_intervalmdp_jl_spec(spec_path, problem::Problem) =
    write_intervalmdp_jl_spec(spec_path, specification(problem))

function intervalmdp_jl_property_dict(prop::FiniteTimeReachability)
    return Dict(
        "type" => "reachability",
        "reach" => reach(prop),
        "time_horizon" => time_horizon(prop),
        "infinite_time" => false,
    )
end

function intervalmdp_jl_property_dict(prop::InfiniteTimeReachability)
    return Dict(
        "type" => "reachability",
        "reach" => reach(prop),
        "eps" => convergence_eps(prop),
        "infinite_time" => true,
    )
end

function intervalmdp_jl_property_dict(prop::FiniteTimeReachAvoid)
    return Dict(
        "type" => "reach-avoid",
        "reach" => reach(prop),
        "avoid" => avoid(prop),
        "time_horizon" => time_horizon(prop),
        "infinite_time" => false,
    )
end

function intervalmdp_jl_property_dict(prop::InfiniteTimeReachAvoid)
    return Dict(
        "type" => "reach-avoid",
        "reach" => reach(prop),
        "avoid" => avoid(prop),
        "eps" => convergence_eps(prop),
        "infinite_time" => true,
    )
end

function intervalmdp_jl_property_dict(prop::FiniteTimeReward)
    return Dict(
        "type" => "reward",
        "reward" => reward(prop),
        "discount" => discount(prop),
        "time_horizon" => time_horizon(prop),
        "infinite_time" => false,
    )
end

function intervalmdp_jl_property_dict(prop::InfiniteTimeReward)
    return Dict(
        "type" => "reward",
        "reward" => reward(prop),
        "discount" => discount(prop),
        "eps" => convergence_eps(prop),
        "infinite_time" => true,
    )
end

function intervalmdp_jl_satisfaction_mode(mode::SatisfactionMode)
    if mode == Optimistic
        return "optimistic"
    else
        return "pessimistic"
    end
end

function intervalmdp_jl_strategy_mode(mode::StrategyMode)
    if mode == Minimize
        return "minimize"
    else
        return "maximize"
    end
end
