# Data storage formats

- ASCII seems to dominate (but is very inefficient)
- Binary formats are more efficient but not standardized

## PRISM
\IMDPjl supports reading and writing [PRISM](https://www.prismmodelchecker.org/) [1] explicit data format. 
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
|-----------------|---------------------------------------------------------------------------------------------------------------|
| First line      | The first line containing symbolically `(v1, v2, ..., vn)` is a list of ``n`` variables in the model.         |
| ``m`` lines where ``m`` is the number of states | Each line contains `i:(v1, v2, ..., vn)` where `i` is the index of the state and `(v1, v2, ..., vn)`` is an assignment of values to the variables in the model. Indices are zero-indexed.                                                     |

#### Labels .lab
| Number of lines | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| First line      | Contains a space-separated list of labels with index `i="label"`. The first two must be `0="init" 1="deadlock"`.         |
| All remaining lines | Contains `i: j1 j2 j3 ...` where `i` is a state index and `j1 j2 j3 ...` are space-separated indices of labels associated with state `i`.                                                    |

#### Transitions .tra
| Number of lines | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| First line      | `num\_states num\_choices num\_transitions` where `num\_state` must match the number in the state file.       |
| Following `num\_transitions` lines | A list of transition probabilities with the format `src\_idx act\_idx dest\_idx [p\_lower,p\_upper] action`.                                                    |

#### Property .pctl
| Number of lines | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
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
|-----------------|---------------------------------------------------------------------------------------------------------------|
| First line      | `num\_states`.                                                                                                |
| Second line     | `num\_actions` (not to be confused with `num\_choices` of PRISM).                                             |
| Third line      | `num\_terminal`.                                                                                              |
| The following `num\_terminal` lines | Indices (zero-indexed) of terminal states, one per line.                                  |
| The following `num\_terminal` lines | Indices (zero-indexed) of terminal states, one per line.                                  |
| All remaining lines | A list of transition probabilities with the format `src\_idx act\_idx dest\_idx p\_lower p\_upper`.       |

## IntervalMDP.jl
An important disadvantage of both PRISM and bmdp-tool data formats is that they store the data in ASCII. As a result, they both require the parsing of floating point numbers and each character requires one byte of storage space. To address this problem, IntervalMDP.jl also supports a different _binary_ format based on NetCDF to store transition probabilities. We use JSON to store the specification, as storage space for the specification is much less a concern, and because JSON is a widely used, human-readable, file format.

```julia
# Read
imdp = read_imdp_jl_model(model_path)
spec = read_imdp_jl_spec(spec_path)
problem = Problem(imdp, spec)

problem = read_imdp_jl(model_path, spec_path)

# Write
write_imdp_jl_model(model_path, imdp_or_problem)
write_imdp_jl_spec(spec_path, spec_or_problem)
```

The new format proposed uses netCDF, which is based on HDF5 underlying, to store transition probabilities, and a JSON file to store the specification. Transition probabilities are stored in CSC-format, which is unfortunately not natively stored in netCDF, nor any widely available format.
Therefore, we store the following attributes and variables in the netCDF file:

+-------------------+---------------------------------------------------------------------------------+
| Global attributes | - `num_states`                                                                  |
|                   | - `model` (either `imc` or `imdp`)                                              |
|                   | - `format` (assert `sparse_csc`)                                                |
|                   | - `rows` (assert `to`)                                                          |
|                   | - `cols` (assert `from` if model is `imc` and `from/action` if model is `imdp`) |
+-------------------+---------------------------------------------------------------------------------+
| Variables         | - `lower\_colptr` (integer)                                                     |
|                   | - `lower\_rowval` (integer)                                                     |
|                   | - `lower\_nzval` (floating point)                                               |
|                   | - `upper\_colptr` (integer)                                                     |
|                   | - `upper\_rowval` (integer)                                                     |
|                   | - `upper\_nzval` (floating point)                                               |
|                   | - `initial\_states` (integer)                                                   |
|                   | - `stateptr` (integer, only for `imdp`)                                         |
|                   | - `action\_vals` (any netCDF supported type, only for `imdp`)                   |
+-------------------+---------------------------------------------------------------------------------+

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