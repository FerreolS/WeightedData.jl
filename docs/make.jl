using WeightedData
using Documenter
using ChainRulesCore
using Measurements
using OnlineSampleStatistics
using RobustModels
using Uncertain

DocMeta.setdocmeta!(WeightedData, :DocTestSetup, :(using WeightedData); recursive = true)

extensions = Module[]
for ext in (:WeightedDataChainRulesCoreExt,
            :WeightedDataMeasurementsExt,
            :WeightedDataOnlineSampleStatisticsExt,
            :WeightedDataRobustModelsExt,
            :WeightedDataUncertainExt)
    m = Base.get_extension(WeightedData, ext)
    m === nothing || push!(extensions, m)
end

const DOC_MODULES = [WeightedData; extensions]

makedocs(;
    modules = DOC_MODULES,
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
