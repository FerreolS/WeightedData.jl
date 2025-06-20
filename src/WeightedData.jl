module WeightedData
using ChainRulesCore
using Measurements

#import Measurements: weightedmean

export likelihood,
    flagbadpix,
    flagbadpix!,
    WeightedArray,
    weightedarray,
    get_data,
    get_precision,
    get_weight,
    ScaledL2Loss,
    weightedmean

include("WeightedArray.jl")
include("likelihood.jl")

end