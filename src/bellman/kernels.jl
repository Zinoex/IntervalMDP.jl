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
    n = supportsize(ambiguity_set)
    Vp_workspace = @view workspace.values_gaps[1:n]
    Vnonzero = @view V[support(ambiguity_set)]
    for (i, (v, p)) in enumerate(zip(Vnonzero, nonzeros(gap(ambiguity_set))))
        Vp_workspace[i] = (v, p)
    end
    gv = sorted_gap_value!(workspace.values_gaps, n, budget, upper_bound, scratch(workspace))
    return dot(V, lower(ambiguity_set)) + gv
end

# Stable sort of `(value, gap)` tuples by value, descending iff `upper_bound`. This is
# exactly `sort!(Vp; rev = upper_bound, by = first, scratch)`: that call builds the
# ordering `ord(isless, first, rev)`, i.e. `By(first, Forward)` or (for `rev = true`)
# `ReverseOrdering(By(first, Forward)) === By(first, Reverse)`, and passes it with the
# default algorithm to `_sort!`. Building the same ordering object in a branch on
# `upper_bound` yields the identical sort call (same algorithm, same ordering, same
# permutation), but with a concrete ordering type: the keyword form constructs the
# ordering from a runtime `Bool` through an unspecialized `by::Function`, which costs a
# dynamic dispatch (and an allocation) per call — dominant for the small supports
# sorted here once per (state, action) pair.
@inline function _sort_by_value!(Vp, upper_bound::Bool, scratch)
    if upper_bound
        sort!(Vp; order = Base.Order.By(first, Base.Order.Reverse), scratch = scratch)
    else
        sort!(Vp; order = Base.Order.By(first, Base.Order.Forward), scratch = scratch)
    end
    return Vp
end
# Any non-`Bool` `upper_bound` keeps the original keyword call (and its behaviour).
@inline _sort_by_value!(Vp, upper_bound, scratch) =
    sort!(Vp; rev = upper_bound, by = first, scratch = scratch)

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

# `sorted_gap_value!(buf, n, budget, upper_bound, scratch)` returns exactly
#
#     Vp = view(buf, 1:n); _sort_by_value!(Vp, upper_bound, scratch); gap_value(Vp, budget)
#
# i.e. the O-max gap mass accumulated along the *stable* sort of the `(value, gap)`
# tuples by value. `gap_value` only reads the sorted prefix up to the element that
# exhausts the budget (typically a third of the support), so the fast path sorts lazily:
# a stable three-way quicksort that recurses into the "better" part first and feeds the
# elements, in sorted order, straight into the identical `gap_value` accumulation,
# stopping as soon as the budget is exhausted. The unsorted remainder is never sorted.
#
# Exactness: a stable sort's output is unique for a given strict weak order, and the
# accumulation consumes elements one by one in that order with the same arithmetic as
# `gap_value`, so the result is bit-identical. Comparisons use the native `<` on the
# values, which coincides with `isless` (what `by = first` sorts with) unless a value is
# `NaN` or `-0.0`; inputs containing either take the original sort-then-accumulate path.
# The contents of `buf` afterwards are unspecified (no caller reads them).
function sorted_gap_value!(
    buf::Vector{Tuple{T, T}},
    n::Integer,
    budget::T,
    upper_bound::Bool,
    scratch::Vector{Tuple{T, T}},
) where {T <: Base.IEEEFloat}
    (n <= length(buf) && n <= length(scratch)) ||
        return _sorted_gap_value_fallback!(buf, n, budget, upper_bound, scratch)
    clean = true
    @inbounds for i in 1:n
        v = buf[i][1]
        clean &= !(isnan(v) | (iszero(v) & signbit(v)))
    end
    clean || return _sorted_gap_value_fallback!(buf, n, budget, upper_bound, scratch)

    res = zero(T)
    if upper_bound
        res, _, _ = _lazy_gap_value!(buf, scratch, 1, Int(n), res, budget, Val(true), 0)
    else
        res, _, _ = _lazy_gap_value!(buf, scratch, 1, Int(n), res, budget, Val(false), 0)
    end
    return res
end

sorted_gap_value!(buf, n, budget, upper_bound, scratch) =
    _sorted_gap_value_fallback!(buf, n, budget, upper_bound, scratch)

function _sorted_gap_value_fallback!(buf, n, budget, upper_bound, scratch)
    Vp = @view buf[1:n]
    _sort_by_value!(Vp, upper_bound, scratch)
    return gap_value(Vp, budget)
end

# Strict "comes first" relation: ascending (`Val(false)`) or descending (`Val(true)`).
@inline _gap_value_lt(::Val{false}, a, b) = a < b
@inline _gap_value_lt(::Val{true}, a, b) = b < a

# One step of `gap_value`'s loop; the flag is its `break` condition.
@inline function _gap_value_step(res::T, budget::T, x::Tuple{T, T}) where {T}
    v, p = x
    p = min(budget, p)
    res += p * v
    budget -= p
    return res, budget, budget <= zero(T)
end

const _LAZY_GAP_VALUE_SMALL = 32
const _LAZY_GAP_VALUE_MAX_DEPTH = 48

# Consume `v[lo:hi]` in stable sorted order into `(res, budget)`. Invariant: `v[lo:hi]`
# holds exactly the elements that follow everything consumed so far in the stable
# order, in their original relative order. Returns `(res, budget, exhausted)`.
function _lazy_gap_value!(
    v::Vector{Tuple{T, T}},
    t::Vector{Tuple{T, T}},
    lo::Int,
    hi::Int,
    res::T,
    budget::T,
    r::Val,
    depth::Int,
) where {T}
    @inbounds while hi - lo >= _LAZY_GAP_VALUE_SMALL
        if depth > _LAZY_GAP_VALUE_MAX_DEPTH
            # Degenerate pivots: stable-sort the rest (same order) and consume it below.
            order = r === Val(true) ? Base.Order.By(first, Base.Order.Reverse) :
                    Base.Order.By(first, Base.Order.Forward)
            sort!(view(v, lo:hi); order = order, scratch = t)
            break
        end

        # Median-of-three pivot value (any element's value is correct; this is speed).
        a = v[lo][1]
        b = v[(lo + hi) >>> 1][1]
        c = v[hi][1]
        pivot = if _gap_value_lt(r, a, b)
            _gap_value_lt(r, b, c) ? b : (_gap_value_lt(r, a, c) ? c : a)
        else
            _gap_value_lt(r, a, c) ? a : (_gap_value_lt(r, b, c) ? c : b)
        end

        # Stable three-way partition: "before" compacted in place at the front, "equal"
        # buffered at the front of `t`, "after" buffered from the back of `t`.
        len = hi - lo + 1
        w = lo
        ne = 0
        na = 0
        for i in lo:hi
            x = v[i]
            if _gap_value_lt(r, x[1], pivot)
                v[w] = x
                w += 1
            elseif _gap_value_lt(r, pivot, x[1])
                t[len - na] = x
                na += 1
            else
                ne += 1
                t[ne] = x
            end
        end
        for j in 1:ne
            v[w + j - 1] = t[j]
        end
        for j in 1:na
            v[w + ne + j - 1] = t[len - j + 1]
        end

        if w > lo
            res, budget, done = _lazy_gap_value!(v, t, lo, w - 1, res, budget, r, depth + 1)
            done && return res, budget, true
        end
        for j in w:(w + ne - 1)
            res, budget, done = _gap_value_step(res, budget, v[j])
            done && return res, budget, true
        end
        lo = w + ne
        depth += 1
    end

    # Small range: stable insertion sort, then consume.
    @inbounds for i in (lo + 1):hi
        x = v[i]
        j = i
        while j > lo
            y = v[j - 1]
            _gap_value_lt(r, x[1], y[1]) || break
            v[j] = y
            j -= 1
        end
        v[j] = x
    end
    @inbounds for j in lo:hi
        res, budget, done = _gap_value_step(res, budget, v[j])
        done && return res, budget, true
    end
    return res, budget, false
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
    n = supportsize(ambiguity_set)
    Vp_workspace = @view workspace.values_gaps[1:n]
    @inbounds for (i, j) in enumerate(support(ambiguity_set))
        Vp_workspace[i] = (V[j], gap(ambiguity_set, j))
    end
    gv = sorted_gap_value!(workspace.values_gaps, n, budget, upper_bound, scratch(workspace))
    return dot(V, lower(ambiguity_set)) + gv
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
    workspace::DenseIntervalOMaxWorkspace,
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

Base.@propagate_inbounds function _populate_actions!(
    workspace::SparseIntervalOMaxWorkspace,
    V,
    model,
    jₛ,
    upper_bound,
)
    marginal = marginals(model)[1]
    actions = available(model, jₛ)
    _populate_actions_shared_order!(workspace, V, marginal, actions, jₛ, upper_bound) &&
        return nothing
    for jₐ in actions
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end
    return nothing
end

# Speed heuristic only (exactness does not depend on it): the shared order pays off when
# many actions share most of their rows (it amortizes one lazy sort over the actions but
# costs a row -> position map per action); with few actions the per-action lazy sort is
# faster (measured: 9 and 25 actions win by ~25%, 3 actions lose).
const _SHARED_ORDER_MIN_ACTIONS = 4
const _SHARED_ORDER_MIN_UNION = 16
const _SHARED_ORDER_MIN_OVERLAP = 2

# Strict total order on `(value, row)`: value ascending (`Rev = false`) or descending
# (`Rev = true`), ties by row ascending. Only used on `NaN`/`-0.0`-free values.
struct _ValueRowOrder{Rev} <: Base.Order.Ordering end
Base.Order.lt(::_ValueRowOrder{false}, a, b) =
    (a[1] < b[1]) | ((a[1] == b[1]) & (a[2] < b[2]))
Base.Order.lt(::_ValueRowOrder{true}, a, b) =
    (b[1] < a[1]) | ((a[1] == b[1]) & (a[2] < b[2]))

# Shared per-state order for the sparse O-max kernel.
#
# Per action, `state_action_bellman` stable-sorts the support's `(V[row], gap)` tuples by
# value — buffered in stored order, so ties keep stored order — and accumulates the gap
# mass along that order (`gap_value`). When a state's actions share most of their
# support rows (e.g. 9 actions over ~1.1k stored entries but only ~140 distinct rows on
# car_parking_S20_I3), it is cheaper to sort the union of the rows once — lazily, only as
# deep as the walks need — by the strict total order "value first (descending iff
# `upper_bound`), then row ascending" and let every action walk that order, skipping rows
# it does not store (a per-target row -> stored-position map, set and reset per action).
#
# Exactness: filtering a total order to an action's rows gives a total order on them;
# when the action's stored rows are strictly ascending (the canonical CSC layout; checked
# per action, otherwise that action takes the per-action kernel), "row ascending" is
# "stored order", so this is exactly the stable sort by value the per-action kernel
# walks. The walk feeds the elements to the identical `gap_value` step with the same
# early exit and adds the same `dot(V, lower)`, so each Q-value is bit-identical. Values
# are compared with `<`/`==`, which agree with `isless` unless a value is `NaN` or `-0.0`;
# then (or for non-`Bool` directions or non-IEEE types) this returns `false` before any
# action is written and the caller runs the per-action kernel. It also declines when the
# state has few actions or they overlap too little (a pure speed heuristic, see below).
_populate_actions_shared_order!(workspace, V, marginal, actions, jₛ, upper_bound) = false

function _populate_actions_shared_order!(
    workspace::SparseIntervalOMaxWorkspace{T},
    V::AbstractVector{T},
    marginal,
    actions,
    jₛ,
    upper_bound::Bool,
) where {T <: Base.IEEEFloat}
    length(actions) >= _SHARED_ORDER_MIN_ACTIONS || return false
    total = 0
    for jₐ in actions
        total += Int(supportsize(marginal[jₐ, jₛ]))
    end
    # Speed heuristic only: share the sort when the union is large enough to be worth
    # it and at most `total / _SHARED_ORDER_MIN_OVERLAP` rows.
    max_union = total ÷ _SHARED_ORDER_MIN_OVERLAP
    max_union >= _SHARED_ORDER_MIN_UNION || return false

    rowpos = workspace.rowpos
    keys = workspace.union_keys
    empty!(keys)

    # Union of the support rows (marking with `rowpos`, then restoring it to zeros).
    @inbounds for jₐ in actions
        for row in support(marginal[jₐ, jₛ])
            if rowpos[row] == 0
                rowpos[row] = 1
                push!(keys, (V[row], Int(row)))
            end
        end
        length(keys) > max_union && break
    end
    clean = true
    @inbounds for (v, row) in keys
        rowpos[row] = 0
        clean &= !(isnan(v) | (iszero(v) & signbit(v)))
    end
    (clean && length(keys) <= max_union) || return false

    if upper_bound
        _shared_order_actions!(workspace, V, marginal, actions, jₛ, _ValueRowOrder{true}())
    else
        _shared_order_actions!(workspace, V, marginal, actions, jₛ, _ValueRowOrder{false}())
    end
    return true
end

# Every action walks `workspace.union_keys` in `order`. The union is sorted lazily, only
# as far as the deepest walk needs: `sorted` is the length of the sorted prefix and
# `segments` a stack of the unsorted remainder's segments, leftmost on top.
function _shared_order_actions!(
    workspace::SparseIntervalOMaxWorkspace{T},
    V,
    marginal,
    actions,
    jₛ,
    order::_ValueRowOrder,
) where {T}
    rowpos = workspace.rowpos
    keys = workspace.union_keys
    nkeys = length(keys)
    t = workspace.union_scratch
    length(t) < nkeys && resize!(t, nkeys)
    segments = workspace.union_segments
    empty!(segments)
    nkeys > 0 && push!(segments, (1, nkeys))
    sorted = 0

    @inbounds for jₐ in actions
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        rows = support(ambiguity_set)
        gaps = nonzeros(gap(ambiguity_set))
        n = length(rows)

        ascending = true
        prev = 0
        for k in 1:n
            row = Int(rows[k])
            ascending &= row > prev
            prev = row
            rowpos[row] = k
        end

        if ascending
            res = zero(T)
            remaining = budget
            pos = 0
            while true
                pos += 1
                if pos > sorted
                    isempty(segments) && break
                    sorted = _extend_sorted_prefix!(keys, t, segments, order)
                end
                v, row = keys[pos]
                k = rowpos[row]
                k == 0 && continue
                res, remaining, exhausted = _gap_value_step(res, remaining, (v, gaps[k]))
                exhausted && break
            end
            for k in 1:n
                rowpos[rows[k]] = 0
            end
            workspace.actions[jₐ] = dot(V, lower(ambiguity_set)) + res
        else
            for k in 1:n
                rowpos[rows[k]] = 0
            end
            workspace.actions[jₐ] =
                state_action_bellman(workspace, V, ambiguity_set, budget, order isa _ValueRowOrder{true})
        end
    end
    return nothing
end

# Sort the leftmost pending segment of `keys` far enough to extend the sorted prefix by
# at least one element (quicksort partitions under the strict total order `order`, which
# has no ties, so stability is moot); returns the new sorted-prefix length.
function _extend_sorted_prefix!(keys, t, segments, order)
    lt = Base.Order.lt
    lo, hi = pop!(segments)
    @inbounds while hi - lo >= _LAZY_GAP_VALUE_SMALL
        a = keys[lo]
        b = keys[(lo + hi) >>> 1]
        c = keys[hi]
        pivot = if lt(order, a, b)
            lt(order, b, c) ? b : (lt(order, a, c) ? c : a)
        else
            lt(order, a, c) ? a : (lt(order, b, c) ? c : b)
        end
        w = lo
        ng = 0
        for i in lo:hi
            x = keys[i]
            if lt(order, x, pivot)
                keys[w] = x
                w += 1
            elseif lt(order, pivot, x)
                ng += 1
                t[ng] = x
            end
        end
        keys[w] = pivot
        for j in 1:ng
            keys[w + j] = t[j]
        end
        ng > 0 && push!(segments, (w + 1, hi))
        push!(segments, (w, w))
        if w > lo
            hi = w - 1
        else
            lo, hi = pop!(segments)
        end
    end
    @inbounds for i in (lo + 1):hi
        x = keys[i]
        j = i
        while j > lo
            y = keys[j - 1]
            lt(order, x, y) || break
            keys[j] = y
            j -= 1
        end
        keys[j] = x
    end
    return hi
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
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, model, ambiguity_sets, budgets, upper_bound)
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
