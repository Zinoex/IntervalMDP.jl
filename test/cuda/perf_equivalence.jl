# CUDA counterpart of test/base/perf_equivalence.jl: the same small case-matrix
# models, moved to the GPU with IntervalMDP.cu / CuModelAdaptor{T}, compared
# against the CPU reference values of the merge base (PerfEquivalence.REFERENCE).
# Float64 within the CPU tolerances, Float32 within 1e-5 (common.md § Julia
# Behavior & Tests). IntervalValueIteration is skipped: it throws on CUDA models
# (Finding B-2 in benchmark/REPORT.md). Product processes are skipped: the merge
# base has no CUDA adapt rule for ProductProcess (IntervalMDP.cu returns the CPU
# product unchanged, and bellman! with a CuArray value function then does scalar
# indexing in src/bellman.jl), matching the empty CUDA eltypes of the product rows in
# benchmark/cases/product.jl. This is Finding B-4 "CUDA coverage gaps" in
# benchmark/REPORT.md § 8 (1a Finding F-2).

@testitem "cuda/perf_equivalence: bellman! O-max" tags = [:cuda] setup = [PerfEquivalence] begin
    using CUDA
    using .PerfEquivalence:
        cases, MODES, REFERENCE, TOL_BELLMAN, TOL_FLOAT32, run_bellman, maxabsdiff, on_gpu
    cuda_ext = Base.get_extension(IntervalMDP, :IntervalMDPCudaExt)

    for (name, c) in cases(), T in (Float64, Float32), (mname, mode) in MODES
        c.product && continue
        @testset "$name $T $mname" begin
            # Adapt is not a test dependency; use the binding of the CUDA extension
            # (IntervalMDP.cu would keep Float64).
            to_dev = x -> cuda_ext.adapt(IntervalMDP.CuModelAdaptor{T}, x)
            @test on_gpu(to_dev(c.model))
            V, ws = run_bellman(c, :default, mode; to_dev, T)
            tol = T == Float64 ? TOL_BELLMAN : TOL_FLOAT32
            @test maxabsdiff(V, REFERENCE["$name/bellman/$mname"]) <= tol
        end
    end
end

@testitem "cuda/perf_equivalence: converged and finite-horizon solves" tags = [:cuda] setup =
    [PerfEquivalence] begin
    using CUDA
    using .PerfEquivalence:
        cases, REFERENCE, TOL_CONVERGED, run_solve, policy_value, maxabsdiff, on_gpu

    rebuild(p::VerificationProblem) =
        VerificationProblem(IntervalMDP.cu(system(p)), specification(p))
    rebuild(p::ControlSynthesisProblem) =
        ControlSynthesisProblem(IntervalMDP.cu(system(p)), specification(p))

    for (name, c) in cases(), (ename, (cpu_problem, alg)) in c.problems
        c.product && continue                       # no CUDA product support (see top)
        alg isa IntervalValueIteration && continue  # Finding B-2: IVI throws on CUDA
        @testset "$name $ename" begin
            problem = rebuild(cpu_problem)
            @test on_gpu(system(problem))
            vals, k, strat, sol = run_solve(problem, alg)
            key = "$name/$ename"
            @test maxabsdiff(vals, REFERENCE["$key/values"]) <= TOL_CONVERGED
            @test abs(k - Int(only(REFERENCE["$key/iterations"]))) <= 1
            if haskey(REFERENCE, "$key/strategy")
                if strat == Int.(REFERENCE["$key/strategy"])
                    @test true
                else
                    @test maxabsdiff(
                        policy_value(problem, alg, sol),
                        REFERENCE["$key/values"],
                    ) <= TOL_CONVERGED
                end
            end
        end
    end
end
