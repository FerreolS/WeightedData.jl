using WeightedData
using Test
using Measurements

@testset "WeightedData.jl" begin

    include("WeightedArray_test.jl")
    include("likelihood_test.jl")
    include("WeightedDataPlotsExt_test.jl")
    include("WeightedDataRobustModelsExt_test.jl")
    include("WeightedDataTwoDimensionalExt_test.jl")

end
