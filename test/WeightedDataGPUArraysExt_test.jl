using GPUArrays
using JLArrays
import WeightedData: value, precision

@testset "WeightedDataGPUArraysExt" begin
    @testset "generic GPU loglikelihood parity" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        model = Float32[2.0, 3.0, 2.5]

        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))
        model_gpu = JLArray(model)

        loss = WeightedData.L2Loss()
        got = loglikelihood(loss, data_gpu, model_gpu)
        ref = sum(0.5f0 .* precisions .* (model .- values) .^ 2)
        @test got ≈ ref rtol = 1f-5 atol = 1f-6
    end

    @testset "shape mismatch throws" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        model_bad = Float32[2.0, 3.0]

        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))
        model_gpu_bad = JLArray(model_bad)

        @test_throws ErrorException("loglikelihood : size(A) != size(model)") loglikelihood(WeightedData.L2Loss(), data_gpu, model_gpu_bad)
    end

    @testset "show for GPU-backed weighted arrays" begin
        data_gpu = WeightedArray(JLArray(Float32[1.0, 2.0]), JLArray(Float32[0.5, 0.2]))
        s = sprint(show, MIME"text/plain"(), data_gpu)
        @test !isempty(s)
        @test occursin("WeightedValue", s) || occursin("WeightedArray", s)
    end
end
