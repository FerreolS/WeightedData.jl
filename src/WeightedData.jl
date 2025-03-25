module WeightedData
using ChainRulesCore

export WeightedPoint,combine,likelihood,scaledlikelihood,gausslikelihood,robustlikelihood,flagbadpix!

include("WeightedPoint.jl")
include("arrays.jl")
include("likelihood.jl")

end
