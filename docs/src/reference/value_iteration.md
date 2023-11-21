# Value iteration

```@docs
value_iteration(problem::Problem{<:IntervalMarkovDecisionProcess, <:AbstractReachability})
```

## O-maximization

```@docs
ominmax(prob, V)
ominmax!(ordering::AbstractStateOrdering, p, prob, V)
partial_ominmax(prob, V, indices)
partial_ominmax!(ordering::AbstractStateOrdering, p, prob, V, indices)
construct_ordering(p::AbstractMatrix)
```