# Tests for the configurable priority-queue sampler (`src/prioritysweeping.jl`).
#
# The sampler is a product of orthogonal choices, so these are organized the
# same way as the trajectory tests: the priority families, the propagation
# rules, the selection policies, the admission rules, then the repair/select
# cycle itself, fairness, the RND priority, and finally end-to-end parity
# against robust value iteration.

@testitem "PrioritizedSweep: state priorities f_P" tags =
    [:base, :priority_sweep, :gsrdp_priority_compute_priority] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    # 2-state/2-action model with hand-verified `_action_uncertainty` values.
    prob1 = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
    prob2 = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
    mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
    vf = (upper = (current = [10.0, 5.0],), lower = (current = [0.0, 5.0],))

    s1, s2 = CartesianIndex(1), CartesianIndex(2)
    f(p, s) = PQ.state_priority(p, s, vf, mdp, nothing)

    @testset "GapPriority is U - L" begin
        @test f(PQ.GapPriority(), s1) ≈ 10.0
        @test f(PQ.GapPriority(), s2) ≈ 0.0
    end

    @testset "BoundPriority reads the bound it is given" begin
        @test f(PQ.BoundPriority(), s1) ≈ 10.0
        @test f(PQ.BoundPriority(), s2) ≈ 5.0
        @test f(PQ.BoundPriority(; bound = IntervalMDP.Lower), s1) ≈ 0.0
        @test f(PQ.BoundPriority(; bound = IntervalMDP.Lower), s2) ≈ 5.0
    end

    @testset "WeightedPriority is f_greedy + beta * f_explore" begin
        # U = [10, 5], gap = [10, 0].
        @test f(PQ.WeightedPriority(0.0), s1) ≈ 10.0
        @test f(PQ.WeightedPriority(0.5), s1) ≈ 15.0
        @test f(PQ.WeightedPriority(1.0), s2) ≈ 5.0
        # beta = 0 reduces exactly to BoundPriority.
        @test f(PQ.WeightedPriority(0.0), s2) == f(PQ.BoundPriority(), s2)
        @test_throws ArgumentError PQ.WeightedPriority(-0.1)
    end

    @testset "ActionUncertaintyPriority" begin
        @test f(PQ.ActionUncertaintyPriority(), s1) ≈ 10.0
        @test f(PQ.ActionUncertaintyPriority(), s2) ≈ 5.0
    end

    @testset "ResidualPriority scores a fixed point at zero" begin
        # Action 1 is a self-loop on state 1, so Q_U(1, 1) = U(1): state 1 is at
        # its own fixed point and the residual is 0, while its gap is 10. This
        # is the discrimination the gap cannot make.
        @test f(PQ.ResidualPriority(), s1) ≈ 0.0
        @test f(PQ.GapPriority(), s1) ≈ 10.0
        # And the residual agrees with the free function it shares code with.
        @test f(PQ.ResidualPriority(), s1) ==
              PQ.bellman_residual_delta(s1, vf, mdp, nothing)
    end

    @testset "compute_priority delegates to the strategy's family" begin
        ss = PQ.PrioritizedSweep(; priority = PQ.GapPriority())
        @test IntervalMDP.compute_priority(ss, s1, vf, mdp, nothing) ≈ 10.0
    end
end

@testitem "PrioritizedSweep: propagation rules" tags =
    [:base, :priority_sweep, :priority_propagate] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    @testset "combine" begin
        @test PQ._combine(PQ.MaxPropagation(), 0.3, 0.7) ≈ 0.7
        @test PQ._combine(PQ.MaxPropagation(), 0.7, 0.3) ≈ 0.7
        @test PQ._combine(PQ.AdditivePropagation(2.0), 0.3, 0.7) ≈ 1.7
        @test PQ._combine(PQ.AdditivePropagation(0.0), 0.3, 0.7) ≈ 0.3
        # Recompute discards the bonus entirely — the behaviour the strategies
        # this module replaces actually had.
        @test PQ._combine(PQ.Recompute(), 0.3, 0.7) ≈ 0.3
        @test_throws ArgumentError PQ.AdditivePropagation(-1.0)
    end

    @testset "MaxPropagation never ranks below f_P alone" begin
        for own in (0.0, 0.25, 1.0), bonus in (0.0, 0.5, 2.0)
            @test PQ._combine(PQ.MaxPropagation(), own, bonus) >= own
        end
    end
end

@testitem "PrioritizedSweep: selection policies" tags =
    [:base, :priority_sweep, :priority_policies] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling
    const TS = IntervalMDP.TrajectorySampling

    values = [1.0, 5.0, 2.0, 4.0]
    fresh = zeros(Int, 4)

    @testset "TopK takes the k highest, in descending order" begin
        @test PQ._select_k(PQ.TopK(), values, fresh, 1) == [2]
        @test PQ._select_k(PQ.TopK(), values, fresh, 2) == [2, 4]
        @test PQ._select_k(PQ.TopK(), values, fresh, 3) == [2, 4, 3]
        # k larger than the state count is clamped, not an error.
        @test length(PQ._select_k(PQ.TopK(), values, fresh, 99)) == 4
    end

    @testset "-Inf is never selected" begin
        masked = [1.0, -Inf, 2.0, -Inf]
        @test PQ._select_k(PQ.TopK(), masked, fresh, 4) == [3, 1]
        @test PQ._select_k(TS.Boltzmann(1.0), masked, fresh, 4) ⊆ [1, 3]
        # Every state masked: an empty batch, not an arbitrary pick.
        @test isempty(PQ._select_k(PQ.TopK(), [-Inf, -Inf], zeros(Int, 2), 2))
    end

    @testset "ties go to the least-recently-selected" begin
        tied = [1.0, 1.0, 1.0]
        # State 2 was selected most recently, state 3 longest ago.
        @test PQ._select_k(PQ.TopK(), tied, [5, 9, 1], 1) == [3]
        @test PQ._select_k(PQ.TopK(), tied, [5, 9, 1], 2) == [3, 1]
    end

    @testset "near-ties rotate too" begin
        # A relative gap below the comparison precision must not starve the
        # state that loses it — the two are treated as tied and rotated.
        near = [1.0, 1.0 + 1e-12, 0.0]
        @test PQ._select_k(PQ.TopK(), near, [9, 1, 0], 1) == [2]
        @test PQ._select_k(PQ.TopK(), near, [1, 9, 0], 1) == [1]
        # A genuine difference is still respected, whatever the recency.
        @test PQ._select_k(PQ.TopK(), [1.0, 2.0, 0.0], [0, 99, 0], 1) == [2]
    end

    @testset "EpsilonGreedy(0) is TopK; EpsilonGreedy(1) covers everything" begin
        @test all(
            _ -> PQ._select_k(TS.EpsilonGreedy(0.0), values, fresh, 2) == [2, 4],
            1:50,
        )
        drawn = Set{Int}()
        for _ in 1:500
            union!(drawn, PQ._select_k(TS.EpsilonGreedy(1.0), values, fresh, 2))
        end
        @test drawn == Set(1:4)
    end

    @testset "a batch is always distinct" begin
        for p in (0.0, 0.3, 1.0), _ in 1:100
            batch = PQ._select_k(TS.EpsilonGreedy(p), values, fresh, 3)
            @test length(unique(batch)) == length(batch)
            @test length(batch) == 3
        end
    end

    @testset "Boltzmann concentrates as T -> 0 and flattens as T -> inf" begin
        @test all(_ -> PQ._select_k(TS.Boltzmann(1e-3), values, fresh, 1) == [2], 1:50)
        drawn = Set{Int}()
        for _ in 1:500
            union!(drawn, PQ._select_k(TS.Boltzmann(1e4), values, fresh, 1))
        end
        @test drawn == Set(1:4)
    end

    @testset "Boltzmann draws k distinct states without replacement" begin
        for _ in 1:100
            batch = PQ._select_k(TS.Boltzmann(1.0), values, fresh, 3)
            @test length(unique(batch)) == 3
        end
    end

    @testset "an unresolved schedule has no temperature to draw with" begin
        decaying = TS.Boltzmann(TS.GapDecayTemperature(0.01, 2.0, 1.0))
        @test_throws ArgumentError PQ._select_k(decaying, values, fresh, 1)
    end
end

@testitem "PrioritizedSweep: admission rules" tags =
    [:base, :priority_sweep, :priority_admit] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    prob = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
    mdp = IntervalMarkovDecisionProcess([prob, prob], [1])
    # gap = [0.5, 0.0]: state 2's bounds have met, state 1's have not.
    vf = (upper = (current = [1.0, 0.25],), lower = (current = [0.5, 0.25],))
    s1, s2 = CartesianIndex(1), CartesianIndex(2)

    @testset "ConvergedSkip declines a state inside the tolerance" begin
        @test PQ.admit(PQ.ConvergedSkip(0.1), s1, vf, mdp, nothing)
        @test !PQ.admit(PQ.ConvergedSkip(0.1), s2, vf, mdp, nothing)
        # The test is `>=`, so a state exactly at the threshold is admitted.
        @test PQ.admit(PQ.ConvergedSkip(0.5), s1, vf, mdp, nothing)
        @test !PQ.admit(PQ.ConvergedSkip(0.6), s1, vf, mdp, nothing)
        @test_throws ArgumentError PQ.ConvergedSkip(-1.0)
    end

    @testset "the default threshold is half the property's tolerance" begin
        spec = Specification(InfiniteTimeReachability([2], 1.0), Pessimistic, Maximize)
        # eps/2 = 0.5, and state 1's gap is exactly 0.5.
        @test PQ.admit(PQ.ConvergedSkip(), s1, vf, mdp, spec)
        @test !PQ.admit(PQ.ConvergedSkip(), s2, vf, mdp, spec)
        # Without a Specification there is no tolerance to derive, so the rule
        # admits everything rather than guessing.
        @test PQ.admit(PQ.ConvergedSkip(), s2, vf, mdp, nothing)
    end

    @testset "ConvergedSkip prunes on the gap, not on f_P" begin
        # `ResidualPriority` scores state 1 at 0 (action 1 self-loops, so it is
        # at its own fixed point) while its gap is 0.5 — a state the termination
        # criterion is still waiting on. Pruning on the priority would drop it.
        @test PQ.state_priority(PQ.ResidualPriority(), s1, vf, mdp, nothing) ≈ 0.0
        @test PQ.admit(PQ.ConvergedSkip(0.1), s1, vf, mdp, nothing)
    end

    @testset "PredicateSkip inverts its predicate" begin
        only_s1 = (s, vf, model, spec) -> s == CartesianIndex(1)
        @test !PQ.admit(PQ.PredicateSkip(only_s1), s1, vf, mdp, nothing)
        @test PQ.admit(PQ.PredicateSkip(only_s1), s2, vf, mdp, nothing)
    end

    @testset "rules AND together: one rejection is enough" begin
        ss = PQ.PrioritizedSweep(;
            admit = [PQ.ConvergedSkip(0.1), PQ.PredicateSkip((s, a, b, c) -> true)],
        )
        @test !PQ._admissible(ss, s1, vf, mdp, nothing)
        @test PQ._admissible(
            PQ.PrioritizedSweep(; admit = PQ.AdmissionRule[]),
            s2,
            vf,
            mdp,
            nothing,
        )
    end
end

@testitem "PrioritizedSweep: repair — full sweep, stale set, early-out, reset" tags =
    [:base, :priority_sweep, :gsrdp_priority_shared] begin
    using IntervalMDP, SparseArrays
    const PQ = IntervalMDP.PriorityQueueSampling

    # 3-state chain, single action per state: 1 -> {1,2}; 2 -> {2,3}; 3 absorbing,
    # so the predecessor relation the stale set follows is 1 <- {1}; 2 <- {1,2};
    # 3 <- {2,3}.
    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    gap_sweep() = PQ.PrioritizedSweep(;
        priority = PQ.GapPriority(),
        propagate = PQ.Recompute(),
        admit = PQ.AdmissionRule[],
    )

    @testset "first call does a full sweep; later calls only touch the stale set" begin
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        ss = gap_sweep()

        # gap = [0.5, 1.0, 0.0] -> state 2 strictly highest, no tie to worry about.
        seq1 = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test seq1 == [CartesianIndex(2)]
        @test ss.priorities[] ≈ [0.5, 1.0, 0.0]
        @test ss.initialized[]
        @test ss.previous_selected[] == [2]

        # Relaxing state 2 makes {1, 2} stale — its predecessors, the states whose
        # own backup reads V(2). State 3 is a *successor* of 2 and must stay stale.
        vf.upper.current[1] = 0.8    # predecessor -> must be picked up
        vf.upper.current[2] = 0.2    # relaxed state itself -> must be picked up
        vf.upper.current[3] = 0.99   # successor only -> must stay stale (cached gap 0.0)

        seq2 = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test seq2 == [CartesianIndex(1)]         # gap 0.8, now the highest
        @test ss.priorities[] ≈ [0.8, 0.2, 0.0]   # state 3 untouched despite the live value change
    end

    @testset "a batch that moved nothing propagates nothing" begin
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        ss = gap_sweep()

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))   # selects state 2
        @test ss.previous_selected[] == [2]

        # Move a *predecessor* of the relaxed state but leave the relaxed state
        # itself alone, so Δ(2) = 0. The repair must skip propagation entirely,
        # leaving state 1's cached priority stale at 0.5 rather than 0.8.
        vf.upper.current[1] = 0.8
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test ss.priorities[] ≈ [0.5, 1.0, 0.0]

        # And the contrast: once the relaxed state does move, the same
        # predecessor is picked up.
        vf.upper.current[2] = 0.2
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test ss.priorities[] ≈ [0.8, 0.2, 0.0]
    end

    @testset "MaxPropagation ranks by the propagated magnitude" begin
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        ss = PQ.PrioritizedSweep(;
            priority = PQ.GapPriority(),
            propagate = PQ.MaxPropagation(),
            admit = PQ.AdmissionRule[],
        )

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))   # selects state 2
        vf.upper.current[2] = 0.2                                    # |Δ(2)| = 0.8

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        # state 1: max(own gap 0.5, 0.7 * 0.8) = 0.56 — the bonus wins, where
        # `Recompute` would have kept 0.5.
        # state 2: max(own gap 0.2, 0.3 * 0.8) = 0.24, the self-loop reading.
        @test ss.priorities[] ≈ [0.56, 0.24, 0.0]
    end

    @testset "reset_sampling_strategy! clears cached state" begin
        ss = PQ.PrioritizedSweep(; priority = PQ.BoundPriority())
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test ss.initialized[]

        IntervalMDP.reset_sampling_strategy!(ss)
        @test !ss.initialized[]
        @test isempty(ss.previous_selected[])
        @test isempty(ss.priorities[])
        @test isempty(ss.selected_snapshot[])
        @test isempty(ss.predecessor_index[])
        @test ss.clock[] == 0
        @test ss.updates[] == 0
    end

    @testset "ties break toward least-recently-selected, not lowest index" begin
        ss = gap_sweep()
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        first = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))[1]
        @test first in (CartesianIndex(1), CartesianIndex(2))
        other = first == CartesianIndex(1) ? CartesianIndex(2) : CartesianIndex(1)

        # Both states 1 and 2 still tie at gap 1.0 (vf untouched), so whichever wasn't
        # just selected must win now — starvation would instead reselect `first` again.
        second = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))[1]
        @test second == other
    end

    @testset "k is clamped to the state count" begin
        ss = PQ.PrioritizedSweep(; k = 99, admit = PQ.AdmissionRule[])
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 0.0],))
        @test length(collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))) == 3
        @test_throws ArgumentError PQ.PrioritizedSweep(; k = 0)
    end

    @testset "an all-converged model yields an empty batch, not a stall" begin
        # Every gap is 0, so `ConvergedSkip` declines everything. GSRDP's own
        # criterion fires on the next check, so an empty batch is the right
        # answer rather than an arbitrary pick.
        ss = PQ.PrioritizedSweep(; admit = [PQ.ConvergedSkip(0.1)])
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [1.0, 1.0, 1.0],))
        @test isempty(collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)))
        @test isempty(collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)))
    end
end

@testitem "PrioritizedSweep: aging is the fairness guarantee" tags =
    [:base, :priority_sweep, :priority_aging] begin
    using IntervalMDP, SparseArrays
    const PQ = IntervalMDP.PriorityQueueSampling

    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    # `BoundPriority` on a value function nothing ever moves: state 1 strictly
    # dominates and is never tied, so the recency tie-break cannot reach it.
    # This is the case the tie-break has no answer for.
    values() = (upper = (current = [1.0, 0.5, 0.0],), lower = (current = [0.0, 0.0, 0.0],))
    sweep(aging) = PQ.PrioritizedSweep(;
        priority = PQ.BoundPriority(),
        propagate = PQ.Recompute(),
        admit = PQ.AdmissionRule[],
        aging = aging,
    )

    @testset "aging = 0 starves a permanently dominated state" begin
        ss, vf = sweep(0.0), values()
        drawn = Set{CartesianIndex}()
        for _ in 1:20
            union!(drawn, collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)))
        end
        @test drawn == Set([CartesianIndex(1)])
    end

    @testset "aging > 0 relaxes every state" begin
        ss, vf = sweep(0.5), values()
        drawn = Set{CartesianIndex}()
        for _ in 1:20
            union!(drawn, collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)))
        end
        @test drawn == Set(CartesianIndex.(1:3))
    end

    @testset "aging does not disturb a genuine ordering while it is fresh" begin
        # The first draw is still the argmax: aging has had no time to accrue.
        ss, vf = sweep(0.5), values()
        @test collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)) ==
              [CartesianIndex(1)]
    end

    @testset "GapPriority is self-correcting without aging" begin
        # A never-relaxed state carries the largest gap in the model, so it
        # cannot be starved even at aging = 0 — which is why that is the default.
        ss = PQ.PrioritizedSweep(; priority = PQ.GapPriority(), admit = PQ.AdmissionRule[])
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 0.0],))
        drawn = Set{CartesianIndex}()
        for _ in 1:20
            union!(drawn, collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing)))
        end
        @test drawn == Set(CartesianIndex.(1:3))
    end

    @testset "aging must be non-negative" begin
        @test_throws ArgumentError PQ.PrioritizedSweep(; aging = -0.1)
    end
end

@testitem "RNDPriority: δ functions" tags = [:base, :priority_sweep, :gsrdp_priority_rnd] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    # Same 2-state/2-action model as the `compute_priority` test above:
    #   state 1: both actions -> state 1
    #   state 2: a1 -> state 1, a2 -> state 2
    prob1 = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
    prob2 = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
    mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
    vf = (upper = (current = [10.0, 5.0],), lower = (current = [0.0, 5.0],))

    @testset "bellman_residual_delta" begin
        # State 1 is a fixed point of its own backup: Q_U(1, ·) = U(1) = 10.
        @test PQ.bellman_residual_delta(CartesianIndex(1), vf, mdp, nothing) ≈ 0.0
        # State 2: maxₐ Q_U(2, a) = max(U(1), U(2)) = 10, against U(2) = 5.
        @test PQ.bellman_residual_delta(CartesianIndex(2), vf, mdp, nothing) ≈ 5.0
    end

    @testset "bellman_residual_delta follows the specification's direction" begin
        minimize = Specification(InfiniteTimeReachability([1], 1e-6), Pessimistic, Minimize)
        # minₐ Q_U(2, a) = min(10, 5) = 5 = U(2), so no residual under Minimize.
        @test PQ.bellman_residual_delta(CartesianIndex(2), vf, mdp, minimize) ≈ 0.0
    end

    @testset "gap_delta / action_uncertainty_delta" begin
        @test PQ.gap_delta(CartesianIndex(1), vf, mdp, nothing) ≈ 10.0
        @test PQ.gap_delta(CartesianIndex(2), vf, mdp, nothing) ≈ 0.0
        @test PQ.action_uncertainty_delta(CartesianIndex(1), vf, mdp, nothing) ≈ 10.0
        @test PQ.action_uncertainty_delta(CartesianIndex(2), vf, mdp, nothing) ≈ 5.0
    end
end

@testitem "RNDPriority: novelty is a backup-recency signal" tags =
    [:base, :priority_sweep, :gsrdp_priority_rnd] begin
    using IntervalMDP, SparseArrays, Random

    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    rnd = IntervalMDP._rnd_construct(mdp; rng = MersenneTwister(42))
    S = collect(IntervalMDP._state_indices(mdp))

    @testset "default features normalise state indices to [-1, 1]" begin
        φ = IntervalMDP._rnd_default_features(mdp)
        @test φ(CartesianIndex(1)) ≈ Float32[-1.0]
        @test φ(CartesianIndex(2)) ≈ Float32[0.0]
        @test φ(CartesianIndex(3)) ≈ Float32[1.0]
    end

    @testset "calibration puts mean novelty at 1 before any training" begin
        IntervalMDP._rnd_calibrate!(rnd, S)
        mean_novelty = sum(IntervalMDP._rnd_novelty(rnd, s) for s in S) / length(S)
        # Exact by construction — the tolerance is for the networks' Float32
        # arithmetic, not for the identity.
        @test mean_novelty ≈ 1.0 rtol = 1e-5
    end

    @testset "a state at the feature-space origin is still novel" begin
        # Regression: with Flux's default zero biases, a state whose features
        # are all zero drives both networks to identical outputs, pinning its
        # novelty at 0 regardless of how little of the space had been swept.
        @test IntervalMDP._rnd_raw_novelty(rnd, CartesianIndex(2)) > 0
    end

    @testset "training on a state suppresses its novelty" begin
        before = IntervalMDP._rnd_novelty(rnd, S[1])
        for _ in 1:200
            IntervalMDP._rnd_train!(rnd, [S[1]])
        end
        @test IntervalMDP._rnd_novelty(rnd, S[1]) < before
    end
end

@testitem "RNDPriority: priority floor and backward propagation" tags =
    [:base, :priority_sweep, :gsrdp_priority_rnd] begin
    using IntervalMDP, SparseArrays, Random
    const PQ = IntervalMDP.PriorityQueueSampling

    # 3-state chain: 1 -> {1,2}; 2 -> {2,3}; 3 absorbing. Predecessor relation
    # (with maxₐ p̄): 1 <- {(1, 0.3)}; 2 <- {(1, 0.7), (2, 0.3)}; 3 <- {(2, 0.7), (3, 1.0)}.
    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    @testset "λ = 0 reduces the priority to δ exactly" begin
        ss = PQ.RNDPriorityQueueSampling(
            1;
            delta = PQ.gap_delta,
            lambda = 0.0,
            rng = MersenneTwister(1),
        )
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        seq = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test seq == [CartesianIndex(2)]
        @test ss.priorities[] ≈ [0.5, 1.0, 0.0]   # the gaps, with no novelty floor
    end

    @testset "the floor lifts a converged but never-relaxed state" begin
        ss = PQ.RNDPriorityQueueSampling(
            1;
            delta = PQ.gap_delta,
            lambda = 100.0,
            rng = MersenneTwister(1),
        )
        # Every gap is ≤ 1, so a λ of 100 against novelty ≈ 1 must dominate δ
        # everywhere — including at state 3, where δ is exactly 0.
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test all(>(1.0), ss.priorities[])
    end

    @testset "the floor drops away once a state has been relaxed" begin
        ss = PQ.RNDPriorityQueueSampling(
            1;
            delta = PQ.gap_delta,
            lambda = 100.0,
            rng = MersenneTwister(1),
        )
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        first = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))[1]
        # Its priority now comes from δ and propagation alone, so it must fall
        # below the floor the never-relaxed states still carry — otherwise the
        # arbitrary novelty ordering would starve them forever.
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        relaxed = LinearIndices(IntervalMDP._state_indices(mdp))[first]
        @test ss.priorities[][relaxed] <= 1.0
    end

    @testset "a relaxed state's Δ propagates to its predecessors" begin
        ss = PQ.RNDPriorityQueueSampling(
            1;
            delta = PQ.gap_delta,
            lambda = 0.0,
            rng = MersenneTwister(1),
        )
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))   # selects state 2
        @test ss.previous_selected[] == [2]
        @test ss.selected_snapshot[] == [(1.0, 0.0)]

        vf.upper.current[2] = 0.2      # |Δ(2)| = 0.8

        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        # priority(1) = maxₐ p̄(2|1,a)·|Δ| = 0.7 · 0.8; priority(2) = max(0.3 · 0.8,
        # its own recomputed gap 0.2); state 3 is not a predecessor of 2, so it
        # keeps its cached 0.0.
        @test ss.priorities[] ≈ [0.56, 0.24, 0.0]
    end

    @testset "reset_sampling_strategy! drops the novelty networks" begin
        ss = PQ.RNDPriorityQueueSampling(1; rng = MersenneTwister(1))
        vf = (upper = (current = [0.5, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test !isnothing(ss.priority.rnd[])

        IntervalMDP.reset_sampling_strategy!(ss)
        @test isnothing(ss.priority.rnd[])
        @test !ss.initialized[]
        @test isempty(ss.previous_selected[])
        @test isempty(ss.selected_snapshot[])
        @test isempty(ss.predecessor_index[])
        @test ss.clock[] == 0
    end

    @testset "invalid hyperparameters are rejected" begin
        @test_throws ArgumentError PQ.RNDPriorityQueueSampling(1; lambda = -1.0)
        @test_throws ArgumentError PQ.RNDPriorityQueueSampling(1; epochs = 0)
    end
end

@testitem "PrioritizedSweep: the backup counter feeds the temperature schedules" tags =
    [:base, :priority_sweep, :priority_temperature] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling
    const TS = IntervalMDP.TrajectorySampling

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    eps = 1e-6
    prop = InfiniteTimeReachability([3], eps)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)

    @testset "sample charges itself exactly what GSRDP will run" begin
        ss = PQ.PrioritizedSweep(; k = 2)
        alg =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        V = IntervalMDP.construct_value_function(alg, problem)
        IntervalMDP._gsrdp_initialize!(V, prop)

        expected = 0
        for _ in 1:5
            seq = IntervalMDP.sample(ss, mdp, nothing, V, spec)
            expected += length(collect(seq)) * (IntervalMDP.num_actions(mdp) + 1)
            @test ss.updates[] == expected
        end
        @test expected > 0

        IntervalMDP.reset_sampling_strategy!(ss)
        @test ss.updates[] == 0
    end

    @testset "the backup counter tracks GSRDP's bellman_updates" begin
        # `n` for UpdateDecayTemperature is meant to BE the solver's backup
        # count, but the strategy has to reconstruct it (`sample` is not handed
        # the iteration). Check the two never drift.
        ss = PQ.PrioritizedSweep(;
            policy = TS.Boltzmann(TS.UpdateDecayTemperature(0.5, 1.0, 0.999)),
            k = 2,
        )
        agree = Ref(true)
        seen = Ref(0)
        callback = function (_, bellman_updates, state_seq)
            state_seq === nothing && return nothing   # the pre-update fire
            seen[] += 1
            agree[] &= (bellman_updates == ss.updates[])
            return nothing
        end

        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = ss,
        )
        solve(problem, gsdp; callback = callback)
        @test seen[] > 0
        @test agree[]
    end

    @testset "a decaying schedule is resolved before every draw" begin
        # An unresolved schedule throws (see the selection-policy tests), so a
        # solve completing at all is the assertion that `sample` resolves it.
        for sched in (
            TS.GapDecayTemperature(0.5, 1.0, 1.0),
            TS.UpdateDecayTemperature(0.5, 1.0, 0.999),
        )
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = PQ.PrioritizedSweep(;
                    policy = TS.Boltzmann(sched),
                    k = 2,
                ),
            )
            (V, _, _) = solve(problem, gsdp)
            @test all(isfinite, V)
        end
    end
end

@testitem "PrioritizedSweep: end-to-end solve() parity with RVI" tags =
    [:base, :priority_sweep, :gsrdp_priority_solve] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling
    const TS = IntervalMDP.TrajectorySampling

    @testset "IMDP verification parity (Pessimistic, Maximize), $N" for N in
                                                                        [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
            upper = N[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = N[1//10 1//5 0; 1//5 1//5 0; 3//10 2//5 1],
            upper = N[1//2 1//2 0; 1//2 2//5 0; 2//5 2//5 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        (V_rvi, _, _) = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)))

        strategies = [
            "default" => PQ.PrioritizedSweep(),
            "k = 2" => PQ.PrioritizedSweep(; k = 2),
            "weighted priority" => PQ.PrioritizedSweep(;
                priority = PQ.WeightedPriority(1.0),
                aging = 1e-3,
            ),
            "bound priority + aging" =>
                PQ.PrioritizedSweep(; priority = PQ.BoundPriority(), aging = 1e-2),
            "residual priority + aging" =>
                PQ.PrioritizedSweep(; priority = PQ.ResidualPriority(), aging = 1e-2),
            "action uncertainty + aging" => PQ.PrioritizedSweep(;
                priority = PQ.ActionUncertaintyPriority(),
                aging = 1e-2,
            ),
            "RND priority" => PQ.PrioritizedSweep(; priority = PQ.RNDPriority(), k = 2),
            "additive propagation" =>
                PQ.PrioritizedSweep(; propagate = PQ.AdditivePropagation(1.0)),
            "recompute propagation" =>
                PQ.PrioritizedSweep(; propagate = PQ.Recompute()),
            "epsilon-greedy selection" =>
                PQ.PrioritizedSweep(; policy = TS.EpsilonGreedy(0.2), k = 2),
            "boltzmann selection" =>
                PQ.PrioritizedSweep(; policy = TS.Boltzmann(0.5), k = 2),
            "no admission rule" => PQ.PrioritizedSweep(; admit = PQ.AdmissionRule[]),
        ]

        @testset "$name" for (name, ss) in strategies
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = ss,
            )
            (V, _, _) = solve(problem, gsdp)
            @test maximum(abs, V_rvi .- V) <= 2 * eps
        end
    end
end

@testitem "PrioritizedSweep: parity on implicit vs explicit sink, and synthesis" tags =
    [:base, :priority_sweep, :priority_sink_solve] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    # The same system written two ways: `explicit_mdp` spells out state 3's
    # absorbing self-loop as a third source column; `implicit_mdp` omits it and
    # lets state 3 be an implicit sink. Both must give the same values.
    prob1 = IntervalAmbiguitySets(;
        lower = Float64[0 1//2; 1//10 3//10; 1//5 1//10],
        upper = Float64[1//2 7//10; 3//5 1//2; 7//10 3//10],
    )
    prob2 = IntervalAmbiguitySets(;
        lower = Float64[1//10 1//5; 1//5 1//5; 3//10 2//5],
        upper = Float64[1//2 1//2; 1//2 2//5; 2//5 2//5],
    )
    sink = IntervalAmbiguitySets(;
        lower = Float64[0 0; 0 0; 1 1],
        upper = Float64[0 0; 0 0; 1 1],
    )

    explicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2, sink], [1])
    implicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])

    eps = 1e-6
    prop = InfiniteTimeReachability([3], eps)
    spec = Specification(prop, Pessimistic, Maximize)

    (V_ref, _, _) = solve(
        VerificationProblem(explicit_mdp, spec),
        RobustValueIteration(default_bellman_algorithm(explicit_mdp)),
    )

    strategies = [
        "default" => PQ.PrioritizedSweep(),
        "k = 2" => PQ.PrioritizedSweep(; k = 2),
        "residual + aging" =>
            PQ.PrioritizedSweep(; priority = PQ.ResidualPriority(), aging = 1e-2),
    ]

    @testset "$name on $label" for (name, ss) in strategies,
        (label, m) in ("explicit" => explicit_mdp, "implicit" => implicit_mdp)

        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(m);
            sampling_strategy = ss,
        )
        (V, _, _) = solve(VerificationProblem(m, spec), gsdp)
        @test maximum(abs, V_ref .- V) <= 1e-4
    end

    @testset "control synthesis runs and agrees with verification" begin
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(implicit_mdp);
            sampling_strategy = PQ.PrioritizedSweep(; k = 2),
        )
        sol = solve(ControlSynthesisProblem(implicit_mdp, spec), gsdp)
        # `checkstrategy` asserts the shape matches `source_shape` and every
        # action is in range — i.e. nothing wrote through the sink.
        @test IntervalMDP.checkstrategy(strategy(sol), implicit_mdp) === nothing
    end
end

@testitem "PrioritizedSweep: the legacy constructors" tags =
    [:base, :priority_sweep, :priority_legacy] begin
    using IntervalMDP
    const PQ = IntervalMDP.PriorityQueueSampling

    # Each one is a `PrioritizedSweep` configured to reproduce what it used to
    # do: `Recompute()`, so the propagated magnitude only decides which
    # priorities are recomputed, and no admission rule. `RNDPriority` is the
    # exception — it always did rank by the propagated magnitude.
    @testset "each maps to the configuration it used to be" begin
        gap = PQ.GapPriorityQueueSampling(3)
        @test gap isa PQ.PrioritizedSweep
        @test gap.priority isa PQ.GapPriority
        @test gap.propagate isa PQ.Recompute
        @test isempty(gap.admit)
        @test gap.k == 3

        upper = PQ.UpperBoundPriorityQueueSampling(1)
        @test upper.priority == PQ.BoundPriority(; bound = IntervalMDP.Upper)
        @test upper.propagate isa PQ.Recompute

        au = PQ.ActionUncertaintyPriorityQueueSampling(1)
        @test au.priority isa PQ.ActionUncertaintyPriority
        @test au.propagate isa PQ.Recompute

        rnd = PQ.RNDPriorityQueueSampling(1; lambda = 2.0)
        @test rnd.priority isa PQ.RNDPriority
        @test rnd.priority.lambda == 2.0
        @test rnd.propagate isa PQ.MaxPropagation
    end

    @testset "hyperparameter validation still reaches the caller" begin
        @test_throws ArgumentError PQ.RNDPriorityQueueSampling(1; lambda = -1.0)
        @test_throws ArgumentError PQ.RNDPriorityQueueSampling(1; epochs = 0)
    end

    @testset "end-to-end parity, as before" begin
        prob = IntervalAmbiguitySets(;
            lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
            upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = [0.1 0.2 0.0; 0.2 0.2 0.0; 0.3 0.4 1.0],
            upper = [0.5 0.5 0.0; 0.5 0.4 0.0; 0.4 0.4 1.0],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        eps = 1e-6
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, _, _) = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)))

        @testset "$(name)" for (name, ss) in [
            "gap" => PQ.GapPriorityQueueSampling(2),
            "upper bound" => PQ.UpperBoundPriorityQueueSampling(2),
            "action uncertainty" => PQ.ActionUncertaintyPriorityQueueSampling(2),
            "RND" => PQ.RNDPriorityQueueSampling(2),
        ]
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = ss,
            )
            (V_gsdp, _, _) = solve(problem, gsdp)
            @test maximum(abs, V_rvi .- V_gsdp) <= 1000 * eps
        end
    end
end
