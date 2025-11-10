"""
    bellman(V, model; upper_bound = false, maximize = true)

Compute robust Bellman update with the value function `V` and the model `model`, e.g. [`IntervalMarkovDecisionProcess`](@ref),
that upper or lower bounds the expectation of the value function `V`.
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed.

### Examples
```jldoctest
using IntervalMDP

prob1 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalAmbiguitySets(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ],
    upper = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ]
)

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

model = IntervalMarkovDecisionProcess(transition_probs, istates)

Vprev = [1.0, 2.0, 3.0]
Vcur = IntervalMDP.bellman(Vprev, model; upper_bound = false)

# output

3-element Vector{Float64}:
 1.7
 2.1
 3.0
```

!!! note
    This function will construct a workspace object, a strategy cache, and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.

"""
function bellman(
    V,
    model,
    alg = default_bellman_algorithm(model);
    upper_bound = false,
    maximize = true,
)
    Vres = similar(V, source_shape(model))

    return bellman!(Vres, V, model, alg; upper_bound = upper_bound, maximize = maximize)
end

"""
    bellman!(workspace, strategy_cache, Vres, V, model; upper_bound = false, maximize = true)

Compute in-place robust Bellman update with the value function `V` and the model `model`, 
e.g. [`IntervalMarkovDecisionProcess`](@ref), that upper or lower bounds the expectation of the value function `V`.
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed. 

The output is constructed in the input `Vres` and returned. The workspace object is also modified,
and depending on the type, the strategy cache may be modified as well. See `construct_workspace`
and `construct_strategy_cache` for more details on how to pre-allocate the workspace and strategy cache.

### Examples

```jldoctest
using IntervalMDP

prob1 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalAmbiguitySets(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ],
    upper = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ]
)

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

model = IntervalMarkovDecisionProcess(transition_probs, istates)

Vprev = [1.0, 2.0, 3.0]
workspace = IntervalMDP.construct_workspace(model)
strategy_cache = IntervalMDP.construct_strategy_cache(model)
Vcur = similar(Vprev)

IntervalMDP.bellman!(workspace, strategy_cache, Vcur, Vprev, model; upper_bound = false, maximize = true)

# output

3-element Vector{Float64}:
 1.7
 2.1
 3.0
```
"""
function bellman! end

function bellman!(
    Vres,
    V,
    model,
    alg = default_bellman_algorithm(model);
    upper_bound = false,
    maximize = true,
)
    workspace = construct_workspace(model, alg)
    strategy_cache = construct_strategy_cache(model)

    return bellman!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function bellman!(
    workspace,
    strategy_cache,
    Vres,
    V,
    model::IntervalMarkovProcess;
    upper_bound = false,
    maximize = true,
)
    return _bellman_helper!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function bellman!(
    workspace::ProductWorkspace,
    strategy_cache,
    Vres,
    V,
    model::ProductProcess;
    upper_bound = false,
    maximize = true,
)
    mp = markov_process(model)
    lf = labelling_function(model)
    dfa = automaton(model)

    return _bellman_helper!(
        workspace,
        strategy_cache,
        Vres,
        V,
        dfa,
        mp,
        lf,
        upper_bound,
        maximize,
    )
end

function _bellman_helper!(
    workspace::ProductWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    dfa::DFA,
    mp::IntervalMarkovProcess,
    lf::DeterministicLabelling,
    upper_bound = false,
    maximize = true,
)
    W = workspace.intermediate_values

    @inbounds for state in dfa
        local_strategy_cache = localize_strategy_cache(strategy_cache, state)

        # Select the value function for the current DFA state
        # according to the appropriate DFA transition function
        map!(W, CartesianIndices(state_values(mp))) do idx
            return V[idx, dfa[state, lf[idx]]]
        end

        # For each state in the product process, compute the Bellman operator
        # for the corresponding Markov process
        bellman!(
            workspace.underlying_workspace,
            local_strategy_cache,
            selectdim(Vres, ndims(Vres), state),
            W,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ProductWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V::AbstractArray{R},
    dfa::DFA,
    mp::IntervalMarkovProcess,
    lf::ProbabilisticLabelling,
    upper_bound = false,
    maximize = true,
) where {R}
    W = workspace.intermediate_values

    @inbounds for state in dfa
        local_strategy_cache = localize_strategy_cache(strategy_cache, state)

        # Select the value function for the current DFA state
        # according to the appropriate DFA transition function
        map!(W, CartesianIndices(state_values(mp))) do idx
            v = zero(R)

            for (label, prob) in enumerate(lf[idx])
                new_dfa_state = dfa[state, label]
                v += prob * V[idx, new_dfa_state]
            end

            return v
        end

        # For each state in the product process, compute the Bellman operator
        # for the corresponding Markov process
        bellman!(
            workspace.underlying_workspace,
            local_strategy_cache,
            selectdim(Vres, ndims(Vres), state),
            W,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function localize_strategy_cache(strategy_cache::NoStrategyCache, dfa_state)
    return strategy_cache
end

function localize_strategy_cache(strategy_cache::TimeVaryingStrategyCache, dfa_state)
    return TimeVaryingStrategyCache(
        selectdim(
            strategy_cache.cur_strategy,
            ndims(strategy_cache.cur_strategy),
            dfa_state,
        ),
    )
end

function localize_strategy_cache(strategy_cache::StationaryStrategyCache, dfa_state)
    return StationaryStrategyCache(
        selectdim(strategy_cache.strategy, ndims(strategy_cache.strategy), dfa_state),
    )
end

function localize_strategy_cache(strategy_cache::ActiveGivenStrategyCache, dfa_state)
    return ActiveGivenStrategyCache(
        selectdim(strategy_cache.strategy, ndims(strategy_cache.strategy), dfa_state),
    )
end

###########################################################################
# O-Maximization-based Bellman operator for IntervalMarkovDecisionProcess #
###########################################################################

# Non-threaded
function _bellman_helper!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]

    for jₛ in CartesianIndices(source_shape(marginal))
        state_bellman!(
            workspace,
            strategy_cache,
            Vres,
            V,
            marginal,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

# Threaded
function _bellman_helper!(
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)

    @inbounds marginal = marginals(model)[1]

    @threadstid tid for jₛ in CartesianIndices(source_shape(marginal))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman!(ws, strategy_cache, Vres, V, marginal, jₛ, upper_bound, maximize)
    end

    return Vres
end

Base.@propagate_inbounds function bellman_precomputation!(
    workspace::Union{DenseIntervalOMaxWorkspace, ThreadedDenseIntervalOMaxWorkspace},
    V,
    upper_bound,
)
    # rev=true for upper bound
    sortperm!(permutation(workspace), V; rev = upper_bound, scratch = scratch(workspace))
end

Base.@propagate_inbounds bellman_precomputation!(
    workspace::Union{SparseIntervalOMaxWorkspace, ThreadedSparseIntervalOMaxWorkspace},
    V,
    upper_bound,
) = nothing

Base.@propagate_inbounds function state_bellman!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    marginal,
    jₛ,
    upper_bound,
    maximize,
)
    for jₐ in CartesianIndices(action_shape(marginal))
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end

    Vres[jₛ] = extract_strategy!(strategy_cache, workspace.actions, jₛ, maximize)
end

Base.@propagate_inbounds function state_bellman!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    marginal,
    jₛ,
    upper_bound,
    maximize,
)
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_set = marginal[jₐ, jₛ]
    budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
    Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::DenseIntervalOMaxWorkspace,
    V,
    ambiguity_set,
    budget,
    upper_bound,
)
    return dot(V, lower(ambiguity_set)) +
           gap_value(V, gap(ambiguity_set), budget, permutation(workspace))
end

Base.@propagate_inbounds function gap_value(
    V::AbstractVector{T},
    gap::VR,
    budget,
    perm,
) where {T, VR <: AbstractVector}
    res = zero(T)

    for i in perm
        p = min(budget, gap[i])
        res += p * V[i]

        budget -= p
        if budget <= zero(T)
            break
        end
    end

    return res
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::SparseIntervalOMaxWorkspace,
    V,
    ambiguity_set,
    budget,
    upper_bound,
)
    Vp_workspace = @view workspace.values_gaps[1:supportsize(ambiguity_set)]
    Vnonzero = @view V[support(ambiguity_set)]
    for (i, (v, p)) in enumerate(zip(Vnonzero, nonzeros(gap(ambiguity_set))))
        Vp_workspace[i] = (v, p)
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))

    return dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)
end

Base.@propagate_inbounds function gap_value(
    Vp::VP,
    budget,
) where {T <: Real, VP <: AbstractVector{<:Tuple{T, T}}}
    res = zero(T)

    for (V, p) in Vp
        p = min(budget, p)
        res += p * V

        budget -= p
        if budget <= zero(T)
            break
        end
    end

    return res
end

##########################################################
# McCormick relaxation-based Bellman operator for fIMDPs #
##########################################################

# Non-threaded
function _bellman_helper!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman!(workspace, strategy_cache, Vres, V, model, jₛ, upper_bound, maximize)
    end

    return Vres
end

# Threaded
function _bellman_helper!(
    workspace::ThreadedFactoredIntervalMcCormickWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman!(ws, strategy_cache, Vres, V, model, jₛ, upper_bound, maximize)
    end

    return Vres
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    for jₐ in CartesianIndices(action_shape(model))
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end

    Vres[jₛ] = extract_strategy!(strategy_cache, workspace.actions, jₛ, maximize)
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
    Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredIntervalMcCormickWorkspace,
    V::AbstractArray{R},
    ambiguity_sets,
    upper_bound,
) where {R}
    V = @view V[map(support, ambiguity_sets)...]

    model = workspace.model
    JuMP.empty!(model)

    # Recursively add McCormick variables and constraints for each ambiguity set
    p, _, _ = mccormick_branch(model, ambiguity_sets)

    if upper_bound
        @objective(model, Max, sum(V[I] * p[I] for I in CartesianIndices(p)))
    else
        @objective(model, Min, sum(V[I] * p[I] for I in CartesianIndices(p)))
    end

    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end

Base.@propagate_inbounds function marginal_lp_constraints(model, ambiguity_set::IntervalAmbiguitySet{R}) where {R}
    p = @variable(model, [1:supportsize(ambiguity_set)])
    p_lower = map(i -> lower(ambiguity_set, i), support(ambiguity_set))
    p_upper = map(i -> upper(ambiguity_set, i), support(ambiguity_set))
    for i in eachindex(p)
        set_lower_bound(p[i], p_lower[i])
        set_upper_bound(p[i], p_upper[i])
    end
    @constraint(model, sum(p) == one(R))

    return p, p_lower, p_upper
end

Base.@propagate_inbounds function mccormick_branch(model, ambiguity_sets)
    if length(ambiguity_sets) == 1
        return marginal_lp_constraints(model, ambiguity_sets[1])
    else
        if length(ambiguity_sets) == 2
            p, p_lower, p_upper = marginal_lp_constraints(model, ambiguity_sets[1])
            q, q_lower, q_upper = marginal_lp_constraints(model, ambiguity_sets[2])
        else
            mid = fld(length(ambiguity_sets), 2) + 1
            p, p_lower, p_upper = mccormick_branch(model, ambiguity_sets[1:mid])
            q, q_lower, q_upper = mccormick_branch(model, ambiguity_sets[(mid + 1):end])
        end

        # McCormick envelopes
        sizes = (size(p)..., size(q)...)
        w = Array{VariableRef}(undef, sizes)
        w_lower = Array{eltype(p_lower)}(undef, sizes)
        w_upper = Array{eltype(p_upper)}(undef, sizes)
        for J in CartesianIndices(q)
            for I in CartesianIndices(p)
                w_lower[I, J] = p_lower[I] * q_lower[J]
                w_upper[I, J] = p_upper[I] * q_upper[J]

                w[I, J] = @variable(
                    model,
                    lower_bound = w_lower[I, J],
                    upper_bound = w_upper[I, J]
                )
                @constraint(
                    model,
                    w[I, J] >=
                    p[I] * q_lower[J] + q[J] * p_lower[I] − p_lower[I] * q_lower[J]
                )
                @constraint(
                    model,
                    w[I, J] >=
                    p[I] * q_upper[J] + q[J] * p_upper[I] − p_upper[I] * q_upper[J]
                )
                @constraint(
                    model,
                    w[I, J] <=
                    p[I] * q_upper[J] + q[J] * p_lower[I] − p_lower[I] * q_upper[J]
                )
                @constraint(
                    model,
                    w[I, J] <=
                    p[I] * q_lower[J] + q[J] * p_upper[I] − p_upper[I] * q_lower[J]
                )
            end
        end
        @constraint(model, sum(w) == one(eltype(p_lower)))

        return w, w_lower, w_upper
    end
end

####################################################
# O-Maximization-based Bellman operator for fIMDPs #
####################################################
function _bellman_helper!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    # For each source state
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    # For each source state
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]

        @inbounds state_bellman!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    for jₐ in CartesianIndices(action_shape(model))
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        budgets =
            ntuple(r -> workspace.budgets[r][sub2ind(marginals(model)[r], jₐ, jₛ)], N)
        workspace.actions[jₐ] = state_action_bellman(
            workspace,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
    end

    Vres[jₛ] = extract_strategy!(strategy_cache, workspace.actions, jₛ, maximize)
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
    budgets = ntuple(r -> workspace.budgets[r][sub2ind(marginals(model)[r], jₐ, jₛ)], N)
    Vres[jₛ] =
        state_action_bellman(workspace, V, model, ambiguity_sets, budgets, upper_bound)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredIntervalOMaxWorkspace,
    V,
    model,
    ambiguity_sets,
    budgets,
    upper_bound,
)
    Vₑ = workspace.expectation_cache
    R = valuetype(model)

    ssize = supportsize.(ambiguity_sets)

    # For each higher-level state in the product space
    for Isparse in CartesianIndices(ssize[2:end])
        I = CartesianIndex(support.(ambiguity_sets[2:end], Tuple(Isparse)))

        # For the first dimension, we need to copy the values from V
        v = orthogonal_inner_bellman!(
            workspace,
            @view(V[:, I]),
            ambiguity_sets[1],
            budgets[1],
            upper_bound,
        )
        Vₑ[1][I[1]] = v

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        for d in 2:(length(ambiguity_sets) - 1)
            if Isparse[d - 1] == ssize[d]
                v = orthogonal_inner_bellman!(
                    workspace,
                    Vₑ[d - 1],
                    ambiguity_sets[d],
                    budgets[d],
                    upper_bound,
                )
                fill!(Vₑ[d - 1], zero(R))
                Vₑ[d][I[d]] = v
            else
                break
            end
        end
    end

    # Last dimension
    v = orthogonal_inner_bellman!(
        workspace,
        Vₑ[end],
        ambiguity_sets[end],
        budgets[end],
        upper_bound,
    )
    fill!(Vₑ[end], zero(R))

    return v
end

Base.@propagate_inbounds function orthogonal_inner_bellman!(
    workspace,
    V,
    ambiguity_set,
    budget,
    upper_bound::Bool,
)
    Vp_workspace = @view workspace.values_gaps[1:supportsize(ambiguity_set)]
    @inbounds for (i, j) in enumerate(support(ambiguity_set))
        Vp_workspace[i] = (V[j], gap(ambiguity_set, j))
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))

    return dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)
end

##########################################################
# Vertex enumeration-based Bellman operator for fIMDPs #
##########################################################

# Non-threaded
function _bellman_helper!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman!(workspace, strategy_cache, Vres, V, model, jₛ, upper_bound, maximize)
    end

    return Vres
end

# Threaded
function _bellman_helper!(
    workspace::ThreadedFactoredVertexIteratorWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman!(ws, strategy_cache, Vres, V, model, jₛ, upper_bound, maximize)
    end

    return Vres
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    for jₐ in CartesianIndices(action_shape(model))
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end

    Vres[jₛ] = extract_strategy!(strategy_cache, workspace.actions, jₛ, maximize)
end

Base.@propagate_inbounds function state_bellman!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
    Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredVertexIteratorWorkspace,
    V::AbstractArray{R},
    ambiguity_sets,
    upper_bound,
) where {R}
    iterators = vertex_generator.(ambiguity_sets, workspace.result_vectors)

    optval = upper_bound ? typemin(R) : typemax(R)
    optfunc = upper_bound ? max : min

    for marginal_vertices in Iterators.product(iterators...)
        v = sum(
            V[I] * prod(r -> marginal_vertices[r][I[r]], eachindex(ambiguity_sets)) for
            I in CartesianIndices(num_target.(ambiguity_sets))
        )
        optval = optfunc(optval, v)
    end

    return optval
end
