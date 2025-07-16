using IntervalMDP, IntervalMDP.Data
using Documenter

push!(LOAD_PATH, "../src/")
DocMeta.setdocmeta!(IntervalMDP, :DocTestSetup, :(using IntervalMDP); recursive = true)

makedocs(;
    modules = [IntervalMDP, IntervalMDP.Data],
    authors = "Frederik Baymler Mathiesen <frederik@baymler.com> and contributors",
    sitename = "IntervalMDP.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://www.baymler.com/IntervalMDP.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Data formats" => "data.md",
        "Theory" => "theory.md",
        "Algorithms" => "algorithms.md",
        "Reference" => Any[
            "Systems" => "reference/systems.md",
            "Specifications" => "reference/specifications.md",
            "Solve Interface" => "reference/solve.md",
            "Data Storage" => "reference/data.md",
        ],
        "Index" => "api.md",
    ],
    doctest = false,
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/Zinoex/IntervalMDP.jl", devbranch = "main")
