"""
    AbstractLabelling

An abstract type for labelling functions.
"""
abstract type AbstractLabelling end
abstract type AbstractSingleStepLabelling <: AbstractLabelling end

struct TimeVaryingLabelling{L <: AbstractSingleStepLabelling} <: AbstractLabelling
    labelling_functions::Vector{L}
end

function check_labelling_function(
    labelling_func::TimeVaryingLabelling,
    state_values,
    num_labels,
)
    for lf in labelling_func.labelling_functions
        check_labelling_function(lf, state_values, num_labels)
    end
end

time_length(lf::TimeVaryingLabelling) = length(lf.labelling_functions)