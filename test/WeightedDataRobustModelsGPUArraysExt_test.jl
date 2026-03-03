using RobustModels
using JLArrays
import WeightedData: value, precision

@testset "WeightedDataRobustModelsGPUArraysExt" begin
    @testset "loglikelihood on JLArray-backed WeightedArray" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        model = Float32[2.0, 3.0, 2.5]

        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))
        model_gpu = JLArray(model)

        loss_l2 = L2Loss()
        got_l2 = loglikelihood(loss_l2, data_gpu, model_gpu)
        ref_l2 = sum(RobustModels.rho.(loss_l2, sqrt.(precisions) .* (model .- values)))
        @test got_l2 ≈ ref_l2 rtol = 1f-5 atol = 1f-6

        loss_huber = HuberLoss()
        got_huber = loglikelihood(loss_huber, data_gpu, model_gpu)
        ref_huber = sum(RobustModels.rho.(loss_huber, sqrt.(precisions) .* (model .- values)))
        @test got_huber ≈ ref_huber rtol = 1f-5 atol = 1f-6
    end

    @testset "dispatch remains valid for non-RobustModels loss" begin
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

        @test_throws ErrorException("loglikelihood : size(A) != size(model)") loglikelihood(L2Loss(), data_gpu, model_gpu_bad)
    end

    @testset "zero precision parity" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.0, 1.5]
        model = Float32[2.0, 3.0, 2.5]

        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))
        model_gpu = JLArray(model)

        loss = L2Loss()
        got = loglikelihood(loss, data_gpu, model_gpu)
        ref = sum(RobustModels.rho.(loss, sqrt.(precisions) .* (model .- values)))
        @test got ≈ ref rtol = 1f-5 atol = 1f-6
    end
end
