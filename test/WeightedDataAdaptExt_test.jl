using Adapt
using JLArrays
import WeightedData: get_value, get_precision

@testset "WeightedDataAdaptExt" begin
    data = WeightedArray(Float32[1.0, 2.0, 3.0], Float32[0.5, 0.2, 1.5])

    @test get_value(data) isa Vector{Float32}
    @test get_precision(data) isa Vector{Float32}

    data_gpu_like = adapt(JLArray, data)
    @test get_value(data_gpu_like) isa JLArray
    @test get_precision(data_gpu_like) isa JLArray
    @test get_value(data_gpu_like) == JLArray(Float32[1.0, 2.0, 3.0])
    @test get_precision(data_gpu_like) == JLArray(Float32[0.5, 0.2, 1.5])

    data_back = adapt(Array, data_gpu_like)
    @test get_value(data_back) == get_value(data)
    @test get_precision(data_back) == get_precision(data)

    # Adapting to an already matching backend should preserve backend/storage kind.
    data_gpu_like_2 = adapt(JLArray, data_gpu_like)
    @test get_value(data_gpu_like_2) isa JLArray
    @test get_precision(data_gpu_like_2) isa JLArray

    data_cpu_2 = adapt(Array, data_back)
    @test get_value(data_cpu_2) isa Vector{Float32}
    @test get_precision(data_cpu_2) isa Vector{Float32}
end
