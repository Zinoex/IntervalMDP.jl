```@meta
CurrentModule = IntervalMDP
```

# IntervalMDP
[IntervalMDP.jl](https://github.com/zinoex/IntervalMDP.jl) is a [Julia](https://julialang.org/) package for modeling
and certifying Interval Markov Decision Processes (IMDPs) via Value Iteration.

IMDPs are a generalization of Markov Decision Processes (MDPs) where the transition probabilities
are represented by intervals instead of point values, to model uncertainty. IMDPs are also frequently
chosen as the model for abstracting the dynamics of a stochastic system, as one may compute upper
and lower bounds on transitioning from one region to another.

The aim of this package is to provide a user-friendly interface to solve value iteration for IMDPs
with great efficiency. Furthermore, it provides methods for accelerating the computation of the
certificate using CUDA hardware. See [Algorithms](@ref) for algorithmic advances that this package
introduces for enabling better use of the available hardware and higher performance.

In addition, the package supports two new subclasses of robust MDPs, namely Orthogonally Decoupled IMDPs (OD-IMDPs), or just Orthogonal IMDPs, and mixtures of Orthogonal IMDPs. These models are designed to be more memory-efficient and computationally efficient than the general IMDP model and in many cases have smaller ambiguity sets, while still being able to represent a wide range of uncertainty. See [Theory](@ref) for more information on these models.

#### Features
- O-maximization and value iteration over IMDPs, OD-IMDPs and mixtures of OD-IMDPs
- Dense and sparse matrix support
- Parametric probability types for customizable precision
- Multithreaded CPU and CUDA-accelerated value iteration
- Data loading and writing in formats by various tools (PRISM, bmdp-tool, IntervalMDP.jl)

!!! info
    Until now, all state-of-the-art tools for IMDPs have been standalone programs. 
    We choose to develop this as a a package to enable better integration with other tools and libraries and improving the extensibility. 

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