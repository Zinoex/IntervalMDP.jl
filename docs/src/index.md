```@meta
CurrentModule = IntervalMDP
```

# IntervalMDP
[IntervalMDP.jl](https://github.com/zinoex/IntervalMDP.jl) is a [Julia](https://julialang.org/) package for modeling
and verifying properties of various subclasses of factored Robust Markov Decision Processes (fRMDPs), in particular
Interval Markov Decision Processes (IMDPs) and factored IMDPs (fIMDPs) via Value Iteration.

RMDPs are an extension of Markov Decision Processes (MDPs) that account for uncertainty in the transition
probabilities, and the factored variant introduces state and action variables such that the transition model
is a product of the transition models of the individual variables, allowing for more compact representations
and efficient algorithms. This package focuses on different subclasses of fRMDPs for which value iteration
can be performed efficiently including Interval Markov Chains (IMCs), IMDPs, orthogonally-decoupled IMDPs
(odIMDPs), and fIMDPs. See [Models](@ref) for more information on these models.

The aim of this package is to provide a user-friendly interface to solve verification and control synthesis
problems for fRMDPs with great efficiency, which includes methods for accelerating the computation using
CUDA hardware, pre-allocation, and other optimization techniques. See [Algorithms](@ref) for choices of the
algorithmic implementation of the Bellman operator; the package aims to provide a sensible default choice
of algorithms, but also allows the user to customize the algorithms to their needs.

!!! info
    For some subclasses of fRMDPs, the Bellman operator cannot be computed exactly, and thus, the provided
    Bellman operators are sound approximations. See [Algorithms](@ref) for more information.

The verification and control synthesis problems supported by this package include minimizing/maximizing
pessimistic/optimistic specifications over properties such as reachability, reach-avoid, safety, (discounted)
reward, and expected hitting times, and over finite and infinite horizons. For more complex properties,
the package supports Deterministic Finite Automata (DFA), with lazy product construction and efficient,
cache-friendly algorithms. See [Specifications](@ref) for more information on the supported specifications.

!!! info
    We use the nomenclature "property" to refer to goal, which defines both how the value function
    is initialized and how it is updated after every Bellman iteration, and "specification" refers to a property
    and whether to minimize or maximize either the lower bound (pessimistic) or the upper bound (optimistic) of
    the value function.

#### Features
- Value iteration over IMCs, IMDPs, odIMDPs, and fIMDPs.
- Plenty of built-in specifications including reachability, safety, reach-avoid, discounted reward, and expected hitting times.
- Support for complex specifications via Deterministic Finite Automata (DFA) with lazy product construction.
- Multithreaded CPU and CUDA-accelerated value iteration.
- Dense and sparse matrix support.
- Parametric probability types (`Float64`, `Float32`, `Rational{BigInt}`) for customizable precision. Note that
  `Rational{BigInt}` is not supported for CUDA acceleration.
- Data loading and writing in formats by various tools (PRISM, bmdp-tool, IntervalMDP.jl).
- Extensible and modular design to allow for custom models, distributed storage and computation, novel specifications,
  and additional Bellman operator and model checking algorithms, and integration with other tools and libraries[^1].

## Installation

This package requires Julia v1.9 or later. Refer to the [official documentation](https://julialang.org/downloads/) on how to install it for your system.

To install `IntervalMDP.jl`, use the following command inside Julia's REPL:

```julia
julia> import Pkg; Pkg.add("IntervalMDP")
```

If you want to use the CUDA extension, you also need to install `CUDA.jl`:
```julia
julia> import Pkg; Pkg.add("CUDA")
```

[^1]: State-of-the-art tools for IMDPs are all standalone programs. We choose to develop this as a a package to enable better integration with other tools and improving the extensibility.