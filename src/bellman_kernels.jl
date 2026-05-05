# Bellman scalar kernels — shared by `bellman_v!` (state_bellman.jl) and
# `bellman_q!` (state_action_bellman.jl).
#
# A "Bellman scalar kernel" computes
#
#     q = opt_{γ ∈ Γ_{s,a}} E_{s' ~ γ}[V_prev(s')]
#
# for a single (s, a) given the workspace, V_prev, the ambiguity set(s),
# any per-(s, a) precomputed budgets, and a direction (`upper_bound`).
# It contains no state-iteration or action-reduction logic. The
# higher-level `bellman_v!` and `bellman_q!` orchestrate sweeps that
# call into these scalar kernels per (s, a).

###############################################################################
# Per-iteration precomputation                                                #
###############################################################################
#
# `bellman_precomputation!(workspace, V, upper_bound)` is run ONCE per Bellman
# iteration before any per-(s, a) call. For dense O-Max we sort the indices
# of `V` by value (rev=true if `upper_bound`) so each scalar call can do an
# O(|nonzero gap|) accumulation against that sorted permutation. For sparse
# O-Max no global precomputation is needed (sort happens per-(s, a) on the
# nonzero subset). Factored workspaces don't have a global precomputation
# either — orthogonality is exploited per (s, a) in the scalar kernel.

Base.@propagate_inbounds function bellman_precomputation!(
    workspace::Union{DenseIntervalOMaxWorkspace, ThreadedDenseIntervalOMaxWorkspace},
    V,
    upper_bound,
)
    sortperm!(permutation(workspace), V; rev = upper_bound, scratch = scratch(workspace))
end

Base.@propagate_inbounds bellman_precomputation!(
    workspace::Union{SparseIntervalOMaxWorkspace, ThreadedSparseIntervalOMaxWorkspace},
    V,
    upper_bound,
) = nothing

# Factored workspaces — no global precomputation.
Base.@propagate_inbounds bellman_precomputation!(
    workspace::Union{
        FactoredIntervalOMaxWorkspace,
        ThreadedFactoredIntervalOMaxWorkspace,
        FactoredIntervalMcCormickWorkspace,
        ThreadedFactoredIntervalMcCormickWorkspace,
        FactoredVertexIteratorWorkspace,
        ThreadedFactoredVertexIteratorWorkspace,
    },
    V,
    upper_bound,
) = nothing

###############################################################################
# Dense / sparse flat-IMDP O-Max scalar kernels                               #
###############################################################################

# Dense O-Max: `V` is a state-shape vector, `ambiguity_set` is one column of
# the (lower, gap) representation, `budget = 1 - sum(lower)`. We assume
# `bellman_precomputation!` has populated `workspace.permutation` with a
# `sortperm` of `V` in the right direction.
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

# Sparse O-Max: only nonzero rows of `gap` are scanned, so we materialize
# (V[i], gap[i]) tuples for the support, sort by V (`rev = upper_bound`),
# and accumulate budget × V along that order.
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

###############################################################################
# Factored O-Max scalar kernel                                                #
###############################################################################
#
# The factored O-Max kernel exploits orthogonality across marginals: it
# computes the joint expectation by sequential per-marginal O-Max passes
# (`orthogonal_inner_bellman!`) over a small `expectation_cache` indexed
# by the higher-order marginals' support.

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

    for Isparse in CartesianIndices(ssize[2:end])
        I = CartesianIndex(support.(ambiguity_sets[2:end], Tuple(Isparse)))

        v = orthogonal_inner_bellman!(
            workspace,
            @view(V[:, I]),
            ambiguity_sets[1],
            budgets[1],
            upper_bound,
        )
        Vₑ[1][I[1]] = v

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
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))
    return dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)
end

###############################################################################
# Factored McCormick scalar kernel                                            #
###############################################################################
#
# Builds a JuMP LP whose variables span the joint distribution `p[I]` over
# the cross-product of the per-marginal supports, with McCormick envelopes
# linearizing each pairwise product. The objective is `Σ V[I] * p[I]`,
# minimized for `upper_bound=false` (worst-case ambiguity) and maximized
# for `upper_bound=true`.

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredIntervalMcCormickWorkspace,
    V::AbstractArray{R},
    ambiguity_sets,
    upper_bound,
) where {R}
    V = @view V[map(support, ambiguity_sets)...]
    model = workspace.model
    JuMP.empty!(model)
    p, _, _ = mccormick_branch(model, ambiguity_sets)

    if upper_bound
        @objective(model, Max, sum(V[I] * p[I] for I in CartesianIndices(p)))
    else
        @objective(model, Min, sum(V[I] * p[I] for I in CartesianIndices(p)))
    end

    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end

Base.@propagate_inbounds function marginal_lp_constraints(
    model,
    ambiguity_set::IntervalAmbiguitySet{R},
) where {R}
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

###############################################################################
# Factored vertex-enumeration scalar kernel                                   #
###############################################################################
#
# Enumerates all vertices of the joint polytope (Cartesian product of the
# per-marginal vertex sets) and returns the optimum of `V · p` over them.
# Exact but exponential — meant as a small-model reference / property-test
# baseline rather than for production use.

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

###############################################################################
# Per-state action-fill helper                                                #
###############################################################################
#
# `_populate_actions!(workspace, V, model, jₛ, upper_bound)` evaluates
# `state_action_bellman` for every action available at state `jₛ` and
# stores the resulting Q values in `workspace.actions`. It is the shared
# "for each action, do one scalar Bellman" loop used inside `bellman_v!`'s
# state-outer sweep before `extract_strategy!` reduces actions → V[s].
# `bellman_q!` does not call this helper because it iterates (a, s) pairs
# directly into Qres without the per-state buffer.

# Flat IMDP — single ambiguity set per (a, s).
Base.@propagate_inbounds function _populate_actions!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    V,
    model,
    jₛ,
    upper_bound,
)
    marginal = marginals(model)[1]
    for jₐ in available(model, jₛ)
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end
end

# Factored O-Max — needs ambiguity sets per marginal and budget per
# marginal.
Base.@propagate_inbounds function _populate_actions!(
    workspace::FactoredIntervalOMaxWorkspace,
    V,
    model,
    jₛ,
    upper_bound,
)
    for jₐ in available(model, jₛ)
        ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        budgets = getindex.(workspace.budgets, inds)
        workspace.actions[jₐ] = state_action_bellman(
            workspace,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
    end
end

# Factored McCormick / Vertex — no precomputed budgets; the LP / vertex
# enumeration sees the bounds directly.
Base.@propagate_inbounds function _populate_actions!(
    workspace::Union{FactoredIntervalMcCormickWorkspace, FactoredVertexIteratorWorkspace},
    V,
    model,
    jₛ,
    upper_bound,
)
    for jₐ in available(model, jₛ)
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end
end
