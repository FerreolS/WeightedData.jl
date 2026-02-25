using WeightedData
using Documenter

DocMeta.setdocmeta!(WeightedData, :DocTestSetup, :(using WeightedData); recursive = true)

makedocs(;
    modules = [WeightedData],
    authors = "Ferréol Soulez <ferreol.soulez@univ-lyon1.fr>",
    sitename = "WeightedData.jl",
    clean = true,
    checkdocs = :exports,
    format = Documenter.HTML(;
        canonical = "https://FerreolS.github.io/WeightedData.jl",
        edit_link = "master",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/FerreolS/WeightedData.jl",
    devbranch = "master",
)
