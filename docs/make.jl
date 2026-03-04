using WeightedData
using Documenter
using ChainRulesCore
using Measurements
using OnlineSampleStatistics
using RobustModels
using Uncertain

DocMeta.setdocmeta!(WeightedData, :DocTestSetup, :(using WeightedData); recursive = true)

extensions = Module[]
for ext in (
        :WeightedDataChainRulesCoreExt,
        :WeightedDataMeasurementsExt,
        :WeightedDataOnlineSampleStatisticsExt,
        :WeightedDataRobustModelsExt,
        :WeightedDataUncertainExt,
    )
    m = Base.get_extension(WeightedData, ext)
    m === nothing || push!(extensions, m)
end

const DOC_MODULES = [WeightedData; extensions]
const DOC_EXTENSION_MODULES = extensions

module_or_empty(sym) = (m = Base.get_extension(WeightedData, sym); m === nothing ? Module[] : [m])
const DOC_EXT_MEASUREMENTS = module_or_empty(:WeightedDataMeasurementsExt)
const DOC_EXT_ONLINESAMPLESTATISTICS = module_or_empty(:WeightedDataOnlineSampleStatisticsExt)
const DOC_EXT_ROBUSTMODELS = module_or_empty(:WeightedDataRobustModelsExt)
const DOC_EXT_UNCERTAIN = module_or_empty(:WeightedDataUncertainExt)

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
