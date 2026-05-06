"""
    AbstractLabelling

An abstract type for labelling functions.
"""
abstract type AbstractLabelling end
abstract type AbstractSingleStepLabelling <: AbstractLabelling end

select_labelling_function(lf::AbstractSingleStepLabelling, k) = lf

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

select_labelling_function(lf::TimeVaryingLabelling, k) =
    lf.labelling_functions[time_length(lf) - k]
