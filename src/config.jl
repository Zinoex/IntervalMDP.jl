struct Config
    sampling_strategy::Union{Nothing, SamplingStrategy}
    term_criteria::Union{Nothing, TerminationCriteria}
end

Config(; sampling_strategy = nothing, term_criteria = nothing) =
    Config(sampling_strategy, term_criteria)
