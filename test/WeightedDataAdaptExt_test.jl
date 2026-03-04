using Adapt
using JLArrays
import WeightedData: get_value, get_precision

@testset "WeightedDataAdaptExt" begin
    data = WeightedArray(Float32[1.0, 2.0, 3.0], Float32[0.5, 0.2, 1.5])

    data_gpu_like = adapt(JLArray, data)
    @test get_value(data_gpu_like) isa JLArray
    @test get_precision(data_gpu_like) isa JLArray
    @test get_value(data_gpu_like) == JLArray(Float32[1.0, 2.0, 3.0])
    @test get_precision(data_gpu_like) == JLArray(Float32[0.5, 0.2, 1.5])

    data_back = adapt(Array, data_gpu_like)
    @test get_value(data_back) == get_value(data)
    @test get_precision(data_back) == get_precision(data)
end
