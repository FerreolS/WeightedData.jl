module WeightedData
using ChainRulesCore

export WeightedPoint,
    combine,
    likelihood,
    scaledlikelihood,
    gausslikelihood,
    robustlikelihood,
    flagbadpix,
    flagbadpix!,
    WeightedArray,
    get_val,
    get_precision

include("WeightedPoint.jl")
include("WeightedArray.jl")
include("likelihood.jl")

end
