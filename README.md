# IntervalMDP.jl - Interval Markov Decision Processes

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://www.baymler.com/IntervalMDP.jl/dev/)
[![Build Status](https://github.com/zinoex/IntervalMDP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/zinoex/IntervalMDP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/Zinoex/IntervalMDP.jl/graph/badge.svg?token=K62S0148BK)](https://codecov.io/gh/Zinoex/IntervalMDP.jl)

IntervalMDP.jl is a Julia package for modeling and certifying Interval Markov Decision Processes (IMDPs) via Value Iteration.

IMDPs are a generalization of Markov Decision Processes (MDPs) where the transition probabilities
are represented by intervals instead of point values, to model uncertainty. IMDPs are also frequently
chosen as the model for abstracting the dynamics of a stochastic system, as one may compute upper
and lower bounds on transitioning from one region to another.

The aim of this package is to provide a user-friendly interface to solve value iteration for IMDPs
with great efficiency. Furthermore, it provides methods for accelerating the computation of the
certificate using CUDA hardware. 

## Features
- Value iteration (Bellman operator via O-maximization)
- Dense and sparse matrix support
- Parametric probability and value types for customizable precision including rationals and floating-point numbers
- Multithreaded CPU and CUDA-accelerated value iteration
- Data loading and writing in formats by various tools (PRISM, bmdp-tool, IMDP.jl)

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

## Usage
Here is an example of how to use the package to solve a finite horizon reachability problem for an Interval Markov Chain (IMC) with 3 states and 1 initial state.
The goal is to compute the maximum pessimistic probability of reaching state 3 within 10 time steps.
```julia
using IntervalMDP

# IMC
prob = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.0
        0.1 0.3 0.0
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0
        0.6 0.5 0.0
        0.7 0.3 1.0
    ],
)

initial_states = [1]  # Initial states are optional
imc = IntervalMarkovChain(prob, initial_states)

target_set = [3]
prop = FiniteTimeReachability(target_set, 10)  # Time steps
spec = Specification(prop, Pessimistic, Maximize)
problem = VerificationProblem(imc, spec)

# Solve
V, k, residual = solve(problem)
```

See [Usage](https://www.baymler.com/IntervalMDP.jl/dev/usage/) for more information about different specifications, using sparse matrices, and CUDA.

## Ecosystem
Building upon IntervalMDP.jl, we are designing an ecosystem of tools, which currently consists of:

- [IntervalMDPAbstractions.jl](https://github.com/Zinoex/IntervalMDPAbstractions.jl) - constructing abstractions of stochastic dynamical systems to verify properties.

## Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the program “IntervalMDP.jl” (GPU-accelerated value iteration for Interval Markov Decision Processes) written by the Frederik Baymler Mathiesen. Fred van Keulen, Dean of Mechanical Engineering.

© 2024, Frederik Baymler Mathiesen, HERALD Lab, TU Delft
