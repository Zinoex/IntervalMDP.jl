abstract type AbstractAvailableActions end
abstract type SingleTimeStepAvailableActions <: AbstractAvailableActions end

# All actions are available at all states
struct AllAvailableActions{M} <: SingleTimeStepAvailableActions
    action_vars::NTuple{M, Int32}
end
function AllAvailableActions(action_vars::NTuple{M, <:Integer}) where {M}
    action_vars_32 = Int32.(action_vars)
    return AllAvailableActions{M}(action_vars_32)
end

available(aa::AllAvailableActions, jₛ) = CartesianIndices(aa.action_vars)
isavailable(::AllAvailableActions, jₛ, a) = true

function check_available_actions(
    aa::AllAvailableActions,
    source_dims,
    state_vars,
    action_vars,
)
    if any(aa.action_vars .!= action_vars)
        throw(
            ArgumentError(
                "AllAvailableActions must have action_vars equal to the MDP's action_vars. Got $(aa.action_vars) vs $action_vars.",
            ),
        )
    end
end

# List of available actions for each state
struct ListAvailableActions{
    N,
    M,
    V <: AbstractVector{CartesianIndex{M}},
    A <: AbstractArray{V},
} <: SingleTimeStepAvailableActions
    states::A
end

function check_available_actions(
    aa::AbstractAvailableActions,
    source_dims,
    state_vars,
    action_vars,
)
    for (d, v, s) in zip(source_dims, state_vars, size(aa.states))
        if !(d <= s <= v)
            throw(
                ArgumentError(
                    "The size of available actions must be between the number of source dims $d and the number of state variables $v. Got $s.",
                ),
            )
        end
    end

    for available_actions in aa.states  # Iterate over each state's available actions
        for a in available_actions
            if !(all(1 .<= a .<= action_vars))
                throw(
                    ArgumentError(
                        "Each action must be between 1 and the number of action variables $action_vars. Got $a.",
                    ),
                )
            end
        end
    end
end
available(aa::ListAvailableActions, jₛ) = aa.states[jₛ]
isavailable(aa::ListAvailableActions, jₛ, a) = a in aa.states[jₛ]

# Time-varying available actions
struct TimeVaryingAvailableActions{A <: SingleTimeStepAvailableActions} <:
       AbstractAvailableActions
    actions::Vector{A}
end

function check_available_actions(
    aa::TimeVaryingAvailableActions,
    source_dims,
    state_vars,
    action_vars,
)
    for single_timestep_aa in aa.actions
        check_available_actions(single_timestep_aa, source_dims, state_vars, action_vars)
    end
end
time_length(aa::TimeVaryingAvailableActions) = length(aa.actions)
