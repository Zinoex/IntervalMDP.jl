```@meta
CurrentModule = IMDP
```

# IMDP
[IMDP](https://github.com/zinoex/IMDP.jl) is a [Julia](https://julialang.org/) package for modeling
and certifying Interval Markov Decision Processes (IMDPs) via Value Iteration.

IMDPs are a generalization of Markov Decision Processes (MDPs) where the transition probabilities
are represented by intervals instead of point values, to model uncertainty. IMDPs are also frequently
chosen as the model for abstracting the dynamics of a stochastic system, as one may compute upper
and lower bounds on transitioning from one region to another.

The aim of this package is to provide a user-friendly interface to solve value iteration for IMDPs
with great efficiency. Furthermore, it provides methods for accelerating the computation of the
certificate using CUDA hardware. See [Algorithms](@ref) for algorithmic advances that this package
introduces for enabling better use of the available hardware and higher performance.

#### Features
- O-maximization and value iteration
- Dense and sparse matrix support
- Parametric probability types for customizable precision
- Multithreaded CPU and CUDA-accelerated value iteration
- Data loading and writing in formats by various tools (PRISM, bmdp-tool, IMDP.jl)

!!! info
    Until now, all state-of-the-art tools for IMDPs have been standalone programs. 
    This is explicitly a package, enabling better integration with other tools and libraries.

## Installation

This package requires Julia v1.9 or later. Refer to the [official documentation](https://julialang.org/downloads/) on how to install it for your system.

To install `IMDP.jl`, use the following command inside Julia's REPL:

```julia
julia> import Pkg; Pkg.add("IMDP")
```

If you want to use the CUDA extension, you also need to install `CUDA.jl`:
```julia
julia> import Pkg; Pkg.add("CUDA")
```