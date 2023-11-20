using IMDP
using Documenter

push!(LOAD_PATH,"../src/")
DocMeta.setdocmeta!(IMDP, :DocTestSetup, :(using IMDP); recursive = true)

makedocs(;
    modules = [IMDP],
    authors = "Frederik Baymler Mathiesen <frederik@baymler.com> and contributors",
    repo = "https://github.com/zinoex/IMDP.jl/blob/{commit}{path}#{line}",
    sitename = "IMDP.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://zinoex.github.io/IMDP.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Usage" => "usage.md",
        "Reference" => "reference.md",
    ],
    doctest=false
)

deploydocs(; repo = "github.com/zinoex/IMDP.jl", devbranch = "main")
