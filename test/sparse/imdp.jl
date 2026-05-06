@testitem "bellman" tags = [:sparse, :bellman] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        prob1 = IntervalAmbiguitySets(;
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 1 // 5 1 // 10]),
            upper = sparse(N[1 // 2 7 // 10; 3 // 5 1 // 2; 7 // 10 3 // 10]),
        )
        prob2 = IntervalAmbiguitySets(;
            lower = sparse(N[1 // 10 1 // 5; 1 // 5 3 // 10; 3 // 10 2 // 5]),
            upper = sparse(N[3 // 5 3 // 5; 1 // 2 1 // 2; 2 // 5 2 // 5]),
        )
        prob3 = IntervalAmbiguitySets(;
            lower = sparse(N[0 0; 0 0; 1 1]),
            upper = sparse(N[0 0; 0 0; 1 1]),
        )
        transition_probs = [prob1, prob2, prob3]
        istates = [1]
        mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
        @test initial_states(mdp) == istates
        mdp = IntervalMarkovDecisionProcess(transition_probs)
        @testset "bellman" begin
            V = N[1, 2, 3]
            Vres = let
                _Vres = Array{eltype(V)}(undef, size(V))
                _ws = IntervalMDP.construct_workspace(mdp)
                _sc = IntervalMDP.construct_strategy_cache(mdp)
                IntervalMDP.bellman_v!(
                    _ws,
                    _sc,
                    IntervalMDP.StateValueArray(_Vres),
                    IntervalMDP.StateValueArray(V),
                    mdp;
                    upper_bound = false,
                    maximize = true,
                )
                _Vres
            end
            @test Vres ≈ N[
                1 // 2 * 1 + 3 // 10 * 2 + 1 // 5 * 3,
                3 // 10 * 1 + 3 // 10 * 2 + 2 // 5 * 3,
                1 * 3,
            ]
            Vres = similar(Vres)
            let
                _ws = IntervalMDP.construct_workspace(mdp)
                _sc = IntervalMDP.construct_strategy_cache(mdp)
                IntervalMDP.bellman_v!(
                    _ws,
                    _sc,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    mdp;
                    upper_bound = false,
                    maximize = true,
                )
                Vres
            end
            @test Vres ≈ N[
                1 // 2 * 1 + 3 // 10 * 2 + 1 // 5 * 3,
                3 // 10 * 1 + 3 // 10 * 2 + 2 // 5 * 3,
                1 * 3,
            ]
        end
    end
end

@testitem "explicit sink state" tags = [:sparse, :explicit_sink_state] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        prob1 = IntervalAmbiguitySets(;
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 1 // 5 1 // 10]),
            upper = sparse(N[1 // 2 7 // 10; 3 // 5 1 // 2; 7 // 10 3 // 10]),
        )
        prob2 = IntervalAmbiguitySets(;
            lower = sparse(N[1 // 10 1 // 5; 1 // 5 3 // 10; 3 // 10 2 // 5]),
            upper = sparse(N[3 // 5 3 // 5; 1 // 2 1 // 2; 2 // 5 2 // 5]),
        )
        prob3 = IntervalAmbiguitySets(;
            lower = sparse(N[0 0; 0 0; 1 1]),
            upper = sparse(N[0 0; 0 0; 1 1]),
        )
        transition_probs = [prob1, prob2, prob3]
        istates = [1]
        mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
        @test initial_states(mdp) == istates
        mdp = IntervalMarkovDecisionProcess(transition_probs)
        @testset "explicit sink state" begin
            transition_prob = IntervalMDP.interval_prob_hcat(transition_probs)
            @test_throws DimensionMismatch IntervalMarkovChain(transition_prob)
            @testset "finite time reachability" begin
                prop = FiniteTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test V_fixed_it1[3] == N(1)
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[3] == N(1)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test V_fixed_it1[3] == N(1)
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[3] == N(1)
            end
            @testset "infinite time reachability" begin
                prop = InfiniteTimeReachability([3], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_conv, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv .>= N(0))
                @test V_conv[3] == N(1)
            end
            @testset "exact time reachability" begin
                prop = ExactTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test all(V_fixed_it1 .<= N(1))
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test all(V_fixed_it1 .<= N(1))
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                prop = ExactTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                prop = FiniteTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
            end
            @testset "finite time reach/avoid" begin
                prop = FiniteTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test all(V_fixed_it1 .<= N(1))
                @test V_fixed_it1[3] == N(1)
                @test V_fixed_it1[2] == N(0)
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[3] == N(1)
                @test V_fixed_it2[2] == N(0)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test all(V_fixed_it1 .<= N(1))
                @test V_fixed_it1[3] == N(1)
                @test V_fixed_it1[2] == N(0)
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[3] == N(1)
                @test V_fixed_it2[2] == N(0)
            end
            @testset "infinite time reach/avoid" begin
                prop = InfiniteTimeReachAvoid([3], [2], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_conv, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv .>= N(0))
                @test all(V_conv .<= N(1))
                @test V_conv[3] == N(1)
                @test V_conv[2] == N(0)
            end
            @testset "exact time reach/avoid" begin
                prop = ExactTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test all(V_fixed_it1 .<= N(1))
                @test V_fixed_it1[2] == N(0)
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[2] == N(0)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                @test V_fixed_it1[2] == N(0)
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                @test V_fixed_it2[2] == N(0)
                prop = ExactTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                prop = FiniteTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
            end
            @testset "finite time reward" begin
                prop = FiniteTimeReward(N[2, 1, 0], N(9 // 10), 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it1, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .>= N(0))
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_fixed_it2, k, _) = solve(problem)
                @test k == 10
                @test all(V_fixed_it1 .<= V_fixed_it2)
            end
            @testset "infinite time reward" begin
                prop = InfiniteTimeReward(N[2, 1, 0], N(9 // 10), N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_conv, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv .>= N(0))
            end
            @testset "expected exit time" begin
                prop = ExpectedExitTime([3], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_conv1, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv1 .>= N(0))
                @test V_conv1[3] == N(0)
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V_conv2, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv1 .<= V_conv2)
                @test V_conv2[3] == N(0)
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_conv1, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv1 .>= N(0))
                @test V_conv1[3] == N(0)
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V_conv2, _, u) = solve(problem)
                @test maximum(u) <= N(1 // 1000000)
                @test all(V_conv1 .<= V_conv2)
                @test V_conv2[3] == N(0)
            end
        end
    end
end

@testitem "implicit sink state" tags = [:sparse, :implicit_sink_state] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        prob1 = IntervalAmbiguitySets(;
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 1 // 5 1 // 10]),
            upper = sparse(N[1 // 2 7 // 10; 3 // 5 1 // 2; 7 // 10 3 // 10]),
        )
        prob2 = IntervalAmbiguitySets(;
            lower = sparse(N[1 // 10 1 // 5; 1 // 5 3 // 10; 3 // 10 2 // 5]),
            upper = sparse(N[3 // 5 3 // 5; 1 // 2 1 // 2; 2 // 5 2 // 5]),
        )
        prob3 = IntervalAmbiguitySets(;
            lower = sparse(N[0 0; 0 0; 1 1]),
            upper = sparse(N[0 0; 0 0; 1 1]),
        )
        transition_probs = [prob1, prob2, prob3]
        istates = [1]
        mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
        @test initial_states(mdp) == istates
        mdp = IntervalMarkovDecisionProcess(transition_probs)
        @testset "implicit sink state" begin
            transition_probs = [prob1, prob2]
            implicit_mdp = IntervalMarkovDecisionProcess(transition_probs)
            @testset "finite time reachability" begin
                prop = FiniteTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "infinite time reachability" begin
                prop = InfiniteTimeReachability([3], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "exact time reachability" begin
                prop = ExactTimeReachability([3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "finite time reach/avoid" begin
                prop = FiniteTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "infinite time reach/avoid" begin
                prop = InfiniteTimeReachAvoid([3], [2], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "exact time reach/avoid" begin
                prop = ExactTimeReachAvoid([3], [2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "finite time reward" begin
                prop = FiniteTimeReward(N[2, 1, 0], N(9 // 10), 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Pessimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
                spec = Specification(prop, Optimistic, Minimize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "infinite time reward" begin
                prop = InfiniteTimeReward(N[2, 1, 0], N(9 // 10), N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
            @testset "expected exit time" begin
                prop = ExpectedExitTime([3], N(1 // 1000000))
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(mdp, spec)
                (V, k, res) = solve(problem)
                problem_implicit = VerificationProblem(implicit_mdp, spec)
                (V_implicit, k_implicit, res_implicit) = solve(problem_implicit)
                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
        end
    end
end
