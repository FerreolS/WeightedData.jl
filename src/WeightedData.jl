module WeightedData
using ChainRulesCore

export WeightedPoint,
    combine,
    likelihood,
    scaledL2loss,
    l2loss,
    flagbadpix,
    flagbadpix!,
    WeightedArray,
    get_data,
    get_precision,
    get_weight

include("WeightedPoint.jl")
include("WeightedArray.jl")
include("likelihood.jl")

end