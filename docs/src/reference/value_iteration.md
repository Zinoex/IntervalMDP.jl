# Value iteration

```@docs
value_iteration(problem::Problem{<:IntervalMarkovChain, <:IMDP.AbstractReachability}; upper_bound = true, discount = 1.0)
value_iteration(problem::Problem{<:IntervalMarkovDecisionProcess, <:IMDP.AbstractReachability}; upper_bound = true, discount = 1.0)
```

## O-maximization

```@docs
ominmax(prob, V; max = true)
ominmax!(ordering::AbstractStateOrdering, p, prob, V; max = true)
partial_ominmax(prob, V, indices; max = true)
partial_ominmax!(ordering::AbstractStateOrdering, p, prob, V, indices; max = true)
construct_ordering(p::AbstractMatrix)
```