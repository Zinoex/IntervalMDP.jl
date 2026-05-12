struct Config
    sampling_strategy::Union{Nothing, Any}
    term_criteria::Union{Nothing, Any}
end

Config(; sampling_strategy = nothing, term_criteria = nothing) =
    Config(sampling_strategy, term_criteria)
