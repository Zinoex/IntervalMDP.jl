# Data formats

```@meta
CurrentModule = IMDP.Data
```

## PRISM
```@docs
write_prism_file(path_without_file_ending, mdp_or_mc, terminal_states)
```

## bmdp-tool
```@docs
read_bmdp_tool_file(path)
write_bmdp_tool_file(path, mdp::IntervalMarkovDecisionProcess, spec::AbstractReachability)
write_bmdp_tool_file(path, mdp::IntervalMarkovDecisionProcess, terminal_states::Vector{<:Integer})
```

## IMDP.jl
```@docs
read_imdp_jl_file(path)
write_imdp_jl_file(path, mdp_or_mc, terminal_states)
```