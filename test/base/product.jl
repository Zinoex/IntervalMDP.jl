@testitem "construction product IMDP/DFA" tags = [:base, :construction_product_imdp_dfa] begin
    using IntervalMDP
    @testset "construction product IMDP/DFA" begin
        T = UInt16[1 3 3; 2 1 3; 3 3 3; 1 1 1]
        delta = TransitionFunction(T)
        istate = Int32(1)
        atomic_props = ["a", "b"]
        dfa = DFA(delta, istate, atomic_props)
        prob1 = IntervalAmbiguitySets(;
            lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
            upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = [0.1 0.2; 0.2 0.3; 0.3 0.4],
            upper = [0.6 0.6; 0.5 0.5; 0.4 0.4],
        )
        transition_probs = [prob1, prob2]
        istates = [Int32(1)]
        mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
        @testset "good case" begin
            map = UInt16[1, 2, 3]
            lf = DeterministicLabelling(map)
            prodIMDP = ProductProcess(mdp, dfa, lf)
            @test markov_process(prodIMDP) == mdp
            @test automaton(prodIMDP) == dfa
            @test labelling_function(prodIMDP) == lf
            io = IOBuffer()
            show(io, MIME("text/plain"), prodIMDP)
            str = String(take!(io))
            @test occursin("ProductProcess", str)
            @test occursin("Underlying process", str)
            @test occursin("Automaton", str)
            @test occursin("DeterministicLabelling{UInt16, Vector{UInt16}}", str)
        end
        @testset "IMDP state labelling func input mismatch" begin
            map = UInt16[1, 2]
            lf = DeterministicLabelling(map)
            @test_throws DimensionMismatch ProductProcess(mdp, dfa, lf)
        end
        @testset "DFA inputs labelling func output mismatch (more output than inputs)" begin
            T = UInt16[1 2 2; 2 1 2]
            delta = TransitionFunction(T)
            istate = Int32(1)
            atomic_props = ["a"]
            dfa = DFA(delta, istate, atomic_props)
            map = UInt16[1, 2, 3]
            lf = DeterministicLabelling(map)
            @test_throws DimensionMismatch ProductProcess(mdp, dfa, lf)
        end
    end
end

@testitem "bellman deterministic labelling" tags = [:base, :bellman_deterministic_labelling] begin
    using IntervalMDP
    @testset "bellman deterministic labelling" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob = IntervalAmbiguitySets(;
                    lower = N[0 5 // 10 0; 1 // 10 3 // 10 0; 2 // 10 1 // 10 1],
                    upper = N[5 // 10 7 // 10 0; 6 // 10 5 // 10 0; 7 // 10 3 // 10 1],
                )
                mc = IntervalMarkovChain(prob)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                labelling = DeterministicLabelling(Int32[1, 1, 2])
                prod_proc = ProductProcess(mc, dfa, labelling)
                V = N[4 1; 2 3; 0 5]
                Vres = let
                    _Qres = Array{eltype(V)}(
                        undef,
                        (IntervalMDP.action_values(prod_proc)..., size(V)...),
                    )
                    _ws = IntervalMDP.construct_workspace(prod_proc)
                    _sc = IntervalMDP.construct_strategy_cache(prod_proc)
                    IntervalMDP.bellman_q!(
                        _ws,
                        _sc,
                        IntervalMDP.StateActionValueArray(_Qres),
                        IntervalMDP.StateValueArray(V),
                        prod_proc;
                        upper_bound = false,
                    )
                    _Qres
                end
                Vtar = N[30 // 10 24 // 10; 33 // 10 2; 5 5]
                Vtar = reshape(Vtar, 1, size(Vtar)...)
                @test Vres ≈ Vtar
            end
        end
    end
end

@testitem "bellman deterministic labelling wtih strategy" tags =
    [:base, :bellman_deterministic_labelling_wtih_strategy] begin
    using IntervalMDP
    @testset "bellman deterministic labelling wtih strategy" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob1 = IntervalAmbiguitySets(;
                    lower = N[0 // 10 4 // 10; 1 // 10 3 // 10; 2 // 10 2 // 10],
                    upper = N[5 // 10 9 // 10; 6 // 10 8 // 10; 7 // 10 7 // 10],
                )
                prob2 = IntervalAmbiguitySets(;
                    lower = N[5 // 10 2 // 10; 3 // 10 3 // 10; 1 // 10 4 // 10],
                    upper = N[7 // 10 3 // 10; 5 // 10 6 // 10; 3 // 10 9 // 10],
                )
                prob3 = IntervalAmbiguitySets(;
                    lower = N[0 0; 0 0; 1 1],
                    upper = N[0 0; 0 0; 1 1],
                )
                transition_probs = [prob1, prob2, prob3]
                mdp = IntervalMarkovDecisionProcess(transition_probs)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                labelling = DeterministicLabelling(Int32[1, 1, 2])
                prod_proc = ProductProcess(mdp, dfa, labelling)
                V = N[4 1; 2 3; 0 5]
                workspace = IntervalMDP.construct_workspace(prod_proc)
                eps = one(N) / N(1000)
                Vtar = N[34 // 10 24 // 10; 36 // 10 32 // 10; 5 5]
                Vres = let
                    _Vres = Array{eltype(V)}(undef, size(V))
                    _ws = IntervalMDP.construct_workspace(prod_proc)
                    _sc = IntervalMDP.construct_strategy_cache(prod_proc)
                    IntervalMDP.bellman_v!(
                        _ws,
                        _sc,
                        IntervalMDP.StateValueArray(_Vres),
                        IntervalMDP.StateValueArray(V),
                        prod_proc;
                        upper_bound = false,
                    )
                    _Vres
                end
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    fill(ntuple((_->begin
                        Int32(0)
                    end), 1), 3, 2),
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    [
                        (Int32(2),) (Int32(1),);
                        (Int32(2),) (Int32(2),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    fill(ntuple((_->begin
                        Int32(0)
                    end), 1), 3, 2),
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    [
                        (Int32(2),) (Int32(1),);
                        (Int32(2),) (Int32(2),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.ActiveGivenStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vres ≈ N[30 // 10 24 // 10; 33 // 10 2; 5 5] atol = eps
            end
        end
    end
end

@testitem "bellman probabilistic labelling" tags = [:base, :bellman_probabilistic_labelling] begin
    using IntervalMDP
    @testset "bellman probabilistic labelling" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob = IntervalAmbiguitySets(;
                    lower = N[0 5 // 10 0; 1 // 10 3 // 10 0; 2 // 10 1 // 10 1],
                    upper = N[5 // 10 7 // 10 0; 6 // 10 5 // 10 0; 7 // 10 3 // 10 1],
                )
                mc = IntervalMarkovChain(prob)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                m = N[9 // 10 7 // 10 1 // 10; 1 // 10 3 // 10 9 // 10]
                labelling = ProbabilisticLabelling(m)
                prod_proc = ProductProcess(mc, dfa, labelling)
                V = N[4 1; 2 3; 0 5]
                Vtar = N[302 // 100 24 // 10; 322 // 100 2; 45 // 10 5]
                Vres = let
                    _Vres = Array{eltype(V)}(undef, size(V))
                    _ws = IntervalMDP.construct_workspace(prod_proc)
                    _sc = IntervalMDP.construct_strategy_cache(prod_proc)
                    IntervalMDP.bellman_v!(
                        _ws,
                        _sc,
                        IntervalMDP.StateValueArray(_Vres),
                        IntervalMDP.StateValueArray(V),
                        prod_proc;
                        upper_bound = false,
                    )
                    _Vres
                end
                @test Vres ≈ Vtar
            end
        end
    end
end

@testitem "bellman probabilistic labelling wtih strategy" tags =
    [:base, :bellman_probabilistic_labelling_wtih_strategy] begin
    using IntervalMDP
    @testset "bellman probabilistic labelling wtih strategy" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob1 = IntervalAmbiguitySets(;
                    lower = N[0 // 10 4 // 10; 1 // 10 3 // 10; 2 // 10 2 // 10],
                    upper = N[5 // 10 9 // 10; 6 // 10 8 // 10; 7 // 10 7 // 10],
                )
                prob2 = IntervalAmbiguitySets(;
                    lower = N[5 // 10 2 // 10; 3 // 10 3 // 10; 1 // 10 4 // 10],
                    upper = N[7 // 10 3 // 10; 5 // 10 6 // 10; 3 // 10 9 // 10],
                )
                prob3 = IntervalAmbiguitySets(;
                    lower = N[0 0; 0 0; 1 1],
                    upper = N[0 0; 0 0; 1 1],
                )
                transition_probs = [prob1, prob2, prob3]
                mdp = IntervalMarkovDecisionProcess(transition_probs)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                m = N[9 // 10 7 // 10 1 // 10; 1 // 10 3 // 10 9 // 10]
                labelling = ProbabilisticLabelling(m)
                prod_proc = ProductProcess(mdp, dfa, labelling)
                V = N[4 1; 2 3; 0 5]
                workspace = IntervalMDP.construct_workspace(prod_proc)
                eps = one(N) / N(1000)
                Vtar = N[33 // 10 24 // 10; 346 // 100 32 // 10; 45 // 10 5]
                Vres = let
                    _Vres = Array{eltype(V)}(undef, size(V))
                    _ws = IntervalMDP.construct_workspace(prod_proc)
                    _sc = IntervalMDP.construct_strategy_cache(prod_proc)
                    IntervalMDP.bellman_v!(
                        _ws,
                        _sc,
                        IntervalMDP.StateValueArray(_Vres),
                        IntervalMDP.StateValueArray(V),
                        prod_proc;
                        upper_bound = false,
                    )
                    _Vres
                end
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    fill(ntuple((_->begin
                        Int32(0)
                    end), 1), 3, 2),
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.TimeVaryingStrategyCache(
                    [
                        (Int32(2),) (Int32(1),);
                        (Int32(2),) (Int32(2),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    fill(ntuple((_->begin
                        Int32(0)
                    end), 1), 3, 2),
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.StationaryStrategyCache(
                    [
                        (Int32(2),) (Int32(1),);
                        (Int32(2),) (Int32(2),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vtar ≈ Vres atol = eps
                strategy_cache = IntervalMDP.ActiveGivenStrategyCache(
                    [
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),);
                        (Int32(1),) (Int32(1),)
                    ],
                )
                Vres = copy(V)
                Vres = IntervalMDP.bellman_v!(
                    workspace,
                    strategy_cache,
                    IntervalMDP.StateValueArray(Vres),
                    IntervalMDP.StateValueArray(V),
                    prod_proc;
                    upper_bound = false,
                )
                @test Vres ≈ N[302 // 100 24 // 10; 322 // 100 2; 45 // 10 5] atol = eps
            end
        end
    end
end

@testitem "value iteration deterministic labelling" tags =
    [:base, :value_iteration_deterministic_labelling] begin
    using IntervalMDP
    @testset "value iteration deterministic labelling" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob1 = IntervalAmbiguitySets(;
                    lower = N[0 // 10 5 // 10; 1 // 10 3 // 10; 2 // 10 1 // 10],
                    upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10],
                )
                prob2 = IntervalAmbiguitySets(;
                    lower = N[1 // 10 2 // 10; 2 // 10 3 // 10; 3 // 10 4 // 10],
                    upper = N[6 // 10 6 // 10; 5 // 10 5 // 10; 4 // 10 4 // 10],
                )
                transition_probs = [prob1, prob2]
                mdp = IntervalMarkovDecisionProcess(transition_probs)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                labelling = DeterministicLabelling(Int32[1, 1, 2])
                prod_proc = ProductProcess(mdp, dfa, labelling)
                @testset "finite time reachability" begin
                    prop = FiniteTimeDFAReachability([2], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_fixed_it1, k, res) = solve(problem)
                    @test all(V_fixed_it1 .>= 0)
                    @test k == 10
                    @test V_fixed_it1[:, 2] == N[1, 1, 1]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_fixed_it1 ≈ V_mc
                    prop = FiniteTimeDFAReachability([2], 11)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = VerificationProblem(prod_proc, spec)
                    (V_fixed_it2, k, res) = solve(problem)
                    @test all(V_fixed_it2 .>= 0)
                    @test k == 11
                    @test V_fixed_it2[:, 2] == N[1, 1, 1]
                    @test all(V_fixed_it2 .>= V_fixed_it1)
                end
                @testset "infinite time reachability" begin
                    prop = InfiniteTimeDFAReachability([2], 0.001)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_conv, k, res) = solve(problem)
                    @test all(V_conv .>= 0)
                    @test maximum(res) <= 0.001
                    @test V_conv[:, 2] == N[1, 1, 1]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_conv ≈ V_mc atol = 0.001
                end
                @testset "finite time safety" begin
                    prop = FiniteTimeDFASafety([2], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_fixed_it1, k, res) = solve(problem)
                    @test all(V_fixed_it1 .>= 0)
                    @test k == 10
                    @test V_fixed_it1[:, 2] == N[0, 0, 0]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_fixed_it1 ≈ V_mc
                    prop = FiniteTimeDFASafety([2], 11)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = VerificationProblem(prod_proc, spec)
                    (V_fixed_it2, k, res) = solve(problem)
                    @test all(V_fixed_it2 .>= 0)
                    @test k == 11
                    @test V_fixed_it2[:, 2] == N[0, 0, 0]
                    @test all(V_fixed_it2 .<= V_fixed_it1)
                end
                @testset "infinite time safety" begin
                    prop = InfiniteTimeDFASafety([2], 0.001)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_conv, k, res) = solve(problem)
                    @test all(V_conv .>= 0)
                    @test maximum(res) <= 0.001
                    @test V_conv[:, 2] == N[0, 0, 0]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_conv ≈ V_mc atol = 0.001
                end
            end
        end
    end
end

@testitem "value iteration probabilistic labelling" tags =
    [:base, :value_iteration_probabilistic_labelling] begin
    using IntervalMDP
    @testset "value iteration probabilistic labelling" begin
        for N in [Float32, Float64, Rational{BigInt}]
            @testset "N = $(N)" begin
                prob1 = IntervalAmbiguitySets(;
                    lower = N[0 // 10 5 // 10; 1 // 10 3 // 10; 2 // 10 1 // 10],
                    upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10],
                )
                prob2 = IntervalAmbiguitySets(;
                    lower = N[1 // 10 2 // 10; 2 // 10 3 // 10; 3 // 10 4 // 10],
                    upper = N[6 // 10 6 // 10; 5 // 10 5 // 10; 4 // 10 4 // 10],
                )
                transition_probs = [prob1, prob2]
                mdp = IntervalMarkovDecisionProcess(transition_probs)
                delta = TransitionFunction(Int32[1 2; 2 2])
                istate = Int32(1)
                atomic_props = ["reach"]
                dfa = DFA(delta, istate, atomic_props)
                m = N[9 // 10 7 // 10 1 // 10; 1 // 10 3 // 10 9 // 10]
                labelling = ProbabilisticLabelling(m)
                prod_proc = ProductProcess(mdp, dfa, labelling)
                eps = one(N) / N(1000)
                @testset "finite time reachability" begin
                    prop = FiniteTimeDFAReachability([2], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_fixed_it1, k, res) = solve(problem)
                    @test all(V_fixed_it1 .>= 0)
                    @test k == 10
                    @test V_fixed_it1[:, 2] == N[1, 1, 1]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_fixed_it1 ≈ V_mc
                    prop = FiniteTimeDFAReachability([2], 11)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = VerificationProblem(prod_proc, spec)
                    (V_fixed_it2, k, res) = solve(problem)
                    @test all(V_fixed_it2 .>= 0)
                    @test k == 11
                    @test V_fixed_it2[:, 2] == N[1, 1, 1]
                    @test all(V_fixed_it2 .>= V_fixed_it1)
                end
                @testset "infinite time reachability" begin
                    prop = InfiniteTimeDFAReachability([2], eps)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_conv, k, res) = solve(problem)
                    @test all(V_conv .>= 0)
                    @test maximum(res) <= eps
                    @test V_conv[:, 2] == N[1, 1, 1]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_conv ≈ V_mc atol = eps
                end
                @testset "finite time safety" begin
                    prop = FiniteTimeDFASafety([2], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_fixed_it1, k, res) = solve(problem)
                    @test all(V_fixed_it1 .>= 0)
                    @test k == 10
                    @test V_fixed_it1[:, 2] == N[0, 0, 0]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_fixed_it1 ≈ V_mc
                    prop = FiniteTimeDFASafety([2], 11)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = VerificationProblem(prod_proc, spec)
                    (V_fixed_it2, k, res) = solve(problem)
                    @test all(V_fixed_it2 .>= 0)
                    @test k == 11
                    @test V_fixed_it2[:, 2] == N[0, 0, 0]
                    @test all(V_fixed_it2 .<= V_fixed_it1)
                end
                @testset "infinite time safety" begin
                    prop = InfiniteTimeDFASafety([2], eps)
                    spec = Specification(prop, Pessimistic, Maximize)
                    problem = ControlSynthesisProblem(prod_proc, spec)
                    (policy, V_conv, k, res) = solve(problem)
                    @test all(V_conv .>= 0)
                    @test maximum(res) <= eps
                    @test V_conv[:, 2] == N[0, 0, 0]
                    problem = VerificationProblem(prod_proc, spec, policy)
                    (V_mc, k, res) = solve(problem)
                    @test V_conv ≈ V_mc atol = eps
                end
            end
        end
    end
end
