using IMDP
using Documenter

push!(LOAD_PATH, "../src/")
DocMeta.setdocmeta!(IMDP, :DocTestSetup, :(using IMDP); recursive = true)

makedocs(; 
    modules = [IMDP],
    authors = "Frederik Baymler Mathiesen <frederik@baymler.com> and contributors",
    sitename = "IMDP.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://zinoex.github.io/IMDP.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Data formats" => "data.md",
        "Algorithms" => "algorithms.md",
        "Reference" => Any[
            "Systems" => "reference/systems.md",
            "Specifications" => "reference/specifications.md",
            "Value Iteration" => "reference/value_iteration.md",
            "Certification & synthesis" => "reference/certification.md",
            "Data Storage" => "reference/data.md",
        ],
        "Index" => "api.md",
    ],
    doctest = false,
)

deploydocs(; repo = "github.com/Zinoex/IMDP.jl", devbranch = "main")
