@testmodule OMaxReference begin
    # Cross-check for the Lean theorems `IntervalMDP.OMax.omax_eq_sSup` / `omax_eq_sInf` (dense,
    # lean/IntervalMDPProofs/OMax.lean) and `IntervalMDP.OMax.omaxSparse_eq_omax` (sparse, same
    # file): the O-maximization Bellman update equals the optimum of the interval LP
    #     max / min  ⟨p, V⟩  s.t.  l ≤ p ≤ u,  ∑ p = 1,
    # solved here by brute force with HiGHS (an external, unverified LP solver).
    using IntervalMDP
    using SparseArrays

    const JuMP = IntervalMDP.JuMP
    const HiGHS = IntervalMDP.HiGHS

    const TOL = 1e-9

    # Optimum of ⟨p, V⟩ over P(l, u) for one column (dense vectors).
    function lp_value(l, u, V, upper_bound)
        model = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(model)
        n = length(V)
        JuMP.@variable(model, l[i] <= p[i = 1:n] <= u[i])
        JuMP.@constraint(model, sum(p) == 1)
        if upper_bound
            JuMP.@objective(model, Max, sum(V[i] * p[i] for i in 1:n))
        else
            JuMP.@objective(model, Min, sum(V[i] * p[i] for i in 1:n))
        end
        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.OPTIMAL
        return JuMP.objective_value(model)
    end

    # Reference Bellman update of an IMDP with `num_actions` actions per state: column
    # `(s - 1) * num_actions + a` is the ambiguity set of action `a` in state `s`
    # (actions vary fastest, as in `IntervalMarkovDecisionProcess(prob, num_actions)`).
    function reference_bellman(lower, upper, num_actions, V, upper_bound, maximize)
        L, U = Matrix(lower), Matrix(upper)
        num_states = size(L, 2) ÷ num_actions
        return map(1:num_states) do s
            vals = [
                lp_value(
                    L[:, (s - 1) * num_actions + a],
                    U[:, (s - 1) * num_actions + a],
                    V,
                    upper_bound,
                ) for a in 1:num_actions
            ]
            maximize ? maximum(vals) : minimum(vals)
        end
    end

    # Compare `IntervalMDP.bellman` with the LP reference for both `upper_bound` values and both
    # `maximize` values (all four satisfaction × strategy modes reach O-max via `upper_bound`).
    # `prob` may have been built positionally, so the reference reads `lower` and `lower + gap`.
    function check(prob, num_actions, V)
        mdp = IntervalMarkovDecisionProcess(prob, num_actions)
        lower = prob.lower
        upper = Matrix(lower) + Matrix(prob.gap)
        ok = true
        for upper_bound in (false, true), maximize in (false, true)
            Vres =
                IntervalMDP.bellman(V, mdp; upper_bound = upper_bound, maximize = maximize)
            Vref = reference_bellman(lower, upper, num_actions, V, upper_bound, maximize)
            ok &= isapprox(Vres, Vref; atol = TOL, rtol = 0)
        end
        return ok
    end

    # Dense and sparse (keyword constructor) versions of the same ambiguity sets.
    dense_and_sparse(lower, upper) = (
        IntervalAmbiguitySets(; lower = lower, upper = upper),
        IntervalAmbiguitySets(; lower = sparse(lower), upper = sparse(upper)),
    )

    # A random valid column of `n` targets with support `supp` (indices outside have l = u = 0).
    function random_column(rng, n, supp)
        l = zeros(n)
        u = zeros(n)
        k = length(supp)
        w = rand(rng, k)
        lsum = rand(rng) # ∑ l < 1
        l[supp] = lsum .* w ./ sum(w)
        g = rand(rng, k)
        # Scale the gaps so that ∑ u ≥ 1 and u ≤ 1.
        g .*= (1 - lsum) / sum(g) * (1 + rand(rng))
        u[supp] = min.(l[supp] .+ g, 1.0)
        if sum(u) < 1  # clipping at 1 can only happen with one support entry
            u[supp] .= 1.0
        end
        return l, u
    end
end

@testitem "base/omax_reference: degenerate interval l = u (budget 0)" setup =
    [OMaxReference] begin
    lower = [0.2 0.0 1.0; 0.3 1.0 0.0; 0.5 0.0 0.0]
    for prob in OMaxReference.dense_and_sparse(lower, copy(lower))
        @test OMaxReference.check(prob, 1, [1.0, 2.0, 3.0])
        @test OMaxReference.check(prob, 1, [3.0, -1.0, 0.5])
    end
    # Two actions per state (6 columns, 3 states).
    lower2 = [0.2 0.0 0.1 0.5 0.0 0.3; 0.3 1.0 0.4 0.5 0.0 0.3; 0.5 0.0 0.5 0.0 1.0 0.4]
    for prob in OMaxReference.dense_and_sparse(lower2, copy(lower2))
        @test OMaxReference.check(prob, 2, [1.0, 2.0, 3.0])
    end
end

@testitem "base/omax_reference: ties in V" setup = [OMaxReference] begin
    lower = [0.1 0.0 0.0 0.2; 0.1 0.2 0.0 0.0; 0.1 0.0 0.3 0.0; 0.1 0.0 0.0 0.1]
    upper = [0.6 0.5 0.4 0.5; 0.6 0.5 0.4 0.5; 0.6 0.5 0.4 0.5; 0.6 0.5 0.4 0.5]
    for prob in OMaxReference.dense_and_sparse(lower, upper),
        V in (
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [1.0, 3.0, 3.0, 0.0],
            [0.5, 0.5, 0.5, 2.0],
        )

        @test OMaxReference.check(prob, 1, V)
    end
    # Two actions per state (8 columns, 4 states).
    lower2, upper2 = hcat(lower, lower[:, end:-1:1]), hcat(upper, upper[:, end:-1:1])
    for prob in OMaxReference.dense_and_sparse(lower2, upper2)
        @test OMaxReference.check(prob, 2, [2.0, 2.0, 1.0, 1.0])
    end
end

@testitem "base/omax_reference: all budget in a single successor" setup = [OMaxReference] begin
    # Column 1: only the gap of target 2 is positive, so the whole budget goes there.
    # Column 2: the budget fits exactly into the gap of the highest-valued target.
    # Column 3: one target has u = 1, the rest l = u = 0 is not stored in the sparse case.
    lower = [0.2 0.1 0.0; 0.0 0.2 0.0; 0.3 0.1 0.0]
    upper = [0.2 0.1 0.0; 0.5 0.8 1.0; 0.3 0.1 0.0]
    for prob in OMaxReference.dense_and_sparse(lower, upper),
        V in ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 5.0, 0.0])

        @test OMaxReference.check(prob, 1, V)
    end
end

@testitem "base/omax_reference: sparse column with an empty gap" setup = [OMaxReference] begin
    using SparseArrays

    # A column with no stored entries at all is not a valid ambiguity set (∑ u = 0 < 1), and the
    # positional constructor requires `lower` and `gap` to share their column structure. So the
    # empty gap is a column whose stored gaps are all explicit zeros (budget 0, l = u on the
    # support). Positional constructor: column 1 stores row 2 only, with gap 0.
    I = [2, 1, 2, 3, 1, 3]
    J = [1, 2, 2, 2, 3, 3]
    lower = sparse(I, J, [1.0, 0.1, 0.2, 0.3, 0.0, 0.5], 3, 3)
    gap = sparse(I, J, [0.0, 0.4, 0.3, 0.2, 0.5, 0.0], 3, 3)
    prob = IntervalAmbiguitySets(lower, gap)
    @test SparseArrays.nnz(prob.gap[:, 1]) == 1
    @test all(iszero, SparseArrays.nonzeros(prob.gap[:, 1]))
    for V in ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0])
        @test OMaxReference.check(prob, 1, V)
    end

    # Keyword constructor: column 1 has stored entries whose gaps are all 0 (l = u on the support).
    lower2 = sparse([0.4 0.1 0.0; 0.0 0.2 0.0; 0.6 0.3 1.0])
    upper2 = sparse([0.4 0.5 0.0; 0.0 0.5 0.0; 0.6 0.5 1.0])
    prob2 = IntervalAmbiguitySets(; lower = lower2, upper = upper2)
    @test all(iszero, SparseArrays.nonzeros(prob2.gap[:, 1]))
    for V in ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0])
        @test OMaxReference.check(prob2, 1, V)
    end
end

@testitem "base/omax_reference: random dense and sparse IMDPs" setup = [OMaxReference] begin
    using Random

    rng = MersenneTwister(20261008)
    for _ in 1:40
        n = rand(rng, 2:5)
        num_actions = rand(rng, 1:2)
        lower = zeros(n, n * num_actions)
        upper = zeros(n, n * num_actions)
        for j in 1:(n * num_actions)
            supp = sort(randperm(rng, n)[1:rand(rng, 1:n)])
            lower[:, j], upper[:, j] = OMaxReference.random_column(rng, n, supp)
        end
        # Values with ties: drawn from a small integer set.
        V = Float64.(rand(rng, 0:2, n))
        for prob in OMaxReference.dense_and_sparse(lower, upper)
            @test OMaxReference.check(prob, num_actions, V)
        end
    end
end
