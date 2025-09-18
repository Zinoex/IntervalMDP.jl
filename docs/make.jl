using IntervalMDP, IntervalMDP.Data
using Documenter, DocumenterCitations

push!(LOAD_PATH, "../src/")
DocMeta.setdocmeta!(IntervalMDP, :DocTestSetup, :(using IntervalMDP); recursive = true)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

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
        "Models" => "models.md",
        "Specifications" => "specifications.md",
        "Algorithms" => "algorithms.md",
        "API reference" => Any[
            "Systems" => "reference/systems.md",
            "Specifications" => "reference/specifications.md",
            "Solve Interface" => "reference/solve.md",
            "Data Storage" => "reference/data.md",
            "Index" => "api.md",
        ],
        "Data formats" => "data.md",
        "References" => "references.md",
    ],
    doctest = false,
    checkdocs = :exports,
    plugins = [bib],
)

deploydocs(; repo = "github.com/Zinoex/IntervalMDP.jl", devbranch = "main")
