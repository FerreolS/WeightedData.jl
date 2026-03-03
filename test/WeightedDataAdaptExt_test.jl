using Adapt
using JLArrays
import WeightedData: value, precision

@testset "WeightedDataAdaptExt" begin
    data = WeightedArray(Float32[1.0, 2.0, 3.0], Float32[0.5, 0.2, 1.5])

    data_gpu_like = adapt(JLArray, data)
    @test value(data_gpu_like) isa JLArray
    @test precision(data_gpu_like) isa JLArray
    @test value(data_gpu_like) == JLArray(Float32[1.0, 2.0, 3.0])
    @test precision(data_gpu_like) == JLArray(Float32[0.5, 0.2, 1.5])

    data_back = adapt(Array, data_gpu_like)
    @test value(data_back) == value(data)
    @test precision(data_back) == precision(data)
end
