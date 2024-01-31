# Data storage formats
IntervalMDP.jl supports reading and writing data in various formats, namely PRISM explicit format, bmdp-tool, and our own format (model and specification).
To justify introducing another standard ([see relevant XKCD](https://xkcd.com/927/)), note that the PRISM explicit format and the bmdp-tool format are all written in ASCII, which is very inefficient in terms of storage space (especially for storing floating point numbers) and parsing time. We propose a binary format for the most storage-intensive part of the data, namely the transition probabilities, and use JSON for the specification, which is human- and machine-readable and widely used.

## PRISM
IntervalMDP.jl supports reading and writing [PRISM](https://www.prismmodelchecker.org/) [1] explicit data format. 
The data format is split into 4 different files, one for the states, one for the labels, one for the transition probabilities, and one for the specification. Therefore, our interface for reading PRISM files takes the path without file ending and adds the appropriate ending to each of the four files.

```julia
# Read
problem = read_prism_file(path_without_file_ending)

# Write
write_prism_file(path_without_file_ending, problem)
```

The problem structure contains both the \gls{imdp} and the specification including whether to synthesize a maximizing or minimizing strategy and whether to use an optimistic or pessimistic adversary.

PRISM uses 4 different ASCII-encoded files to store the explicit representation of the system: '.sta' (states), '.lab' (labels), '.tra' (transitions), and '.pctl' (property). In the tables below, we list the format for each file.
The extended details of the PRISM explicit file format can be found in [the appendix of the PRISM manual](https://www.prismmodelchecker.org/manual/Appendices/ExplicitModelFiles).

#### States .sta
| Number of lines | Description                                                                                                   |
|:----------------|:--------------------------------------------------------------------------------------------------------------|
| First line      | The first line containing symbolically `(v1, v2, ..., vn)` is a list of ``n`` variables in the model.         |
| ``m`` lines where ``m`` is the number of states | Each line contains `i:(v1, v2, ..., vn)` where `i` is the index of the state and `(v1, v2, ..., vn)`` is an assignment of values to the variables in the model. Indices are zero-indexed.                                                     |

#### Labels .lab
| Number of lines | Description                                                                                                   |
|:----------------|:--------------------------------------------------------------------------------------------------------------|
| First line      | Contains a space-separated list of labels with index `i="label"`. The first two must be `0="init" 1="deadlock"`.         |
| All remaining lines | Contains `i: j1 j2 j3 ...` where `i` is a state index and `j1 j2 j3 ...` are space-separated indices of labels associated with state `i`.                                                    |

#### Transitions .tra
| Number of lines | Description                                                                                                   |
|:----------------|:--------------------------------------------------------------------------------------------------------------|
| First line      | `num_states num_choices num_transitions` where `num_state` must match the number in the state file.           |
| Following `num_transitions` lines | A list of transition probabilities with the format `src_idx act_idx dest_idx [p_lower,p_upper] action`.                                                    |

#### Property .pctl
| Number of lines | Description                                                                                                   |
|:----------------|:--------------------------------------------------------------------------------------------------------------|
| First line      | PRISM property specification                                                                                  |

## bmdp-tool
[bmdp-tool](https://github.com/aria-systems-group/bmdp-tool) data format is similar to the PRISM explicit format transition probability files, where transition probabilities are stored line-by-line with source, action, destination, and probability bounds in ASCII. Key differences include no explicit listing of states, the fact that it only supports reachability properties, and that terminal states are listed directly in the transition probability file. As a result, bmdp-tool data format is a single file. This format lacks information about whether the reachability is finite or infinite time, and hence the reader only returns the set of terminal states.

```julia
# Read
imdp, terminal_states = read_bmdp_tool_file(path)

# Write
write_bmdp_tool_file(path, problem)
```

bmdp-tool uses only one ASCII file with the following format:

| Number of lines | Description                                                                                                   |
|:----------------|:--------------------------------------------------------------------------------------------------------------|
| First line      | `num_states`.                                                                                                 |
| Second line     | `num_actions` (not to be confused with `num_choices` of PRISM).                                               |
| Third line      | `num_terminal`.                                                                                               |
| The following `num_terminal` lines | Indices (zero-indexed) of terminal states, one per line.                                   |
| The following `num_terminal` lines | Indices (zero-indexed) of terminal states, one per line.                                   |
| All remaining lines | A list of transition probabilities with the format `src_idx act_idx dest_idx p_lower p_upper`.            |

!!! terminology "Choices vs actions"

    In PRISM, the number of choices which is listed in the transition file is the sum of the number of feasible actions in each state. In bmdp-tool, the number of actions is the total number of different actions in the model, i.e. in each state up to `num_actions` may be feasible. This is a subtle difference, but it is important to be aware of as the parsing in either tool requires the right number to be specified.

## IntervalMDP.jl
IntervalMDP.jl also supports a different _binary_ format based on NetCDF to store transition probabilities. We use JSON to store the specification, as storage space for the specification is much less a concern, and because JSON is a widely used, human-readable, file format.

```julia
# Read
imdp = read_intervalmdp_jl_model(model_path)
spec = read_intervalmdp_jl_spec(spec_path)
problem = Problem(imdp, spec)

problem = read_intervalmdp_jl(model_path, spec_path)

# Write
write_intervalmdp_jl_model(model_path, imdp_or_problem)
write_intervalmdp_jl_spec(spec_path, spec_or_problem)
```

The new format proposed uses netCDF, which is based on HDF5 underlying, to store transition probabilities, and a JSON file to store the specification. Transition probabilities are stored in CSC-format, which is unfortunately not natively stored in netCDF, nor any widely available format.
Therefore, we store the following attributes and variables in the netCDF file:

__Global attributes:__

- `num_states`
- `model` (either `imc` or `imdp`)
- `format` (assert `sparse_csc`)
- `rows` (assert `to`)
- `cols` (assert `from` if model is `imc` and `from/action` if model is `imdp`)

__Variables:__
- `lower_colptr` (integer)
- `lower_rowval` (integer)
- `lower_nzval` (floating point)
- `upper_colptr` (integer)
- `upper_rowval` (integer)
- `upper_nzval` (floating point)
- `initial_states` (integer)
- `stateptr` (integer, only for `imdp`)
- `action_vals` (any netCDF supported type, only for `imdp`)

We store the specification in a JSON format where the structure depends on the type of specification.
For a reachability-like specification, the specification is the following format

```json
{
    "property": {
        "type": <"reachability"|"reach-avoid">,
        "infinite_time": <true|false>,
        "time_horizon": <positive int>,
        "eps": <positive float>,
        "reach": [<state_index:positive int>],
        "avoid": [<state_index:positive int>]
    },
    "satisfaction_mode": <"pessimistic"|"optimistic">,
    "strategy_mode": <"minimize"|"maximize">
}
```

For a finite horizon property, `eps` is excluded, and similarly for an infinite horizon property, `time\_horizon` is excluded. 
For a proper reachability property, the `avoid`-field is excluded.

If we instead want to optimize a reward, the format is the following

```json
{
    "property": {
        "type": "reward",
        "infinite_time": <true|false>,
        "time_horizon": <positive int>,
        "eps": <positive float>,
        "reward": [<reward_per_state_index:float>]
        "discount" <float:0-1>
    },
    "satisfaction_mode": <"pessimistic"|"optimistic">,
    "strategy_mode": <"minimize"|"maximize">
}
```

[1] Kwiatkowska, Marta, Gethin Norman, and David Parker. "PRISM 4.0: Verification of probabilistic real-time systems." Computer Aided Verification: 23rd International Conference, CAV 2011, Snowbird, UT, USA, July 14-20, 2011. Proceedings 23. Springer Berlin Heidelberg, 2011.