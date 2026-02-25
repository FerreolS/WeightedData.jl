module WeightedData
using ZippedArrays

import TypeUtils


export WeightedValue,
    weightedmean,
    likelihood,
    flagbadpix,
    flagbadpix!,
    WeightedArray,
    get_value,
    get_precision,
    get_weight,
    ScaledL2Loss

include("WeightedValue.jl")
include("WeightedArray.jl")
include("likelihood.jl")

end
