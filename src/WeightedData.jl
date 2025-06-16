module WeightedData
using ChainRulesCore

export WeightedPoint,
    weightedmean,
    likelihood,
    flagbadpix,
    flagbadpix!,
    WeightedArray,
    get_data,
    get_precision,
    get_weight,
    ScaledL2Loss

include("WeightedPoint.jl")
include("WeightedArray.jl")
include("likelihood.jl")

end