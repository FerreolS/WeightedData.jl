using GPUArrays
using JLArrays
import WeightedData: get_value, get_precision

@testset "WeightedDataGPUArraysExt" begin
    @testset "WeightedArray methods on WeightedArrayGPU" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))

        @test size(data_gpu) == (3,)
        @test get_value(data_gpu) == JLArray(values)
        @test get_precision(data_gpu) == JLArray(precisions)
        @test propertynames(data_gpu) == (:value, :precision)
        @test data_gpu.value == JLArray(values)
        @test data_gpu.precision == JLArray(precisions)

        reshaped = reshape(data_gpu, 3, 1)
        @test size(reshaped) == (3, 1)
        @test get_value(reshaped) == reshape(JLArray(values), 3, 1)
        @test get_precision(reshaped) == reshape(JLArray(precisions), 3, 1)

        shifted = data_gpu + 2.0f0
        @test get_value(shifted) == JLArray(Float32[3.0, 4.0, 5.0])
        @test get_precision(shifted) == JLArray(precisions)

        scaled = 2.0f0 * data_gpu
        @test get_value(scaled) == JLArray(Float32[2.0, 4.0, 6.0])
        @test get_precision(scaled) ≈ JLArray(Float32[0.125, 0.05, 0.375])
    end

    @testset "generic GPU loglikelihood parity" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        model = Float32[2.0, 3.0, 2.5]
        data = WeightedArray(values, precisions)

        data_gpu = WeightedArray(JLArray(values), JLArray(precisions))
        model_gpu = JLArray(model)

        loss = WeightedData.L2Loss()
        got = loglikelihood(loss, data_gpu, model_gpu)
        ref = sum(0.5f0 .* precisions .* (model .- values) .^ 2)
        @test got ≈ ref rtol = 1.0f-5 atol = 1.0f-6

        using Zygote
        lkl, grad = Zygote.withgradient(model -> loglikelihood(loss, data, model), model)

        lkl_gpu, grad_gpu = Zygote.withgradient(model -> loglikelihood(loss, data_gpu, model), model_gpu)

        @test lkl_gpu ≈ lkl rtol = 1.0f-5 atol = 1.0f-6
        @test grad_gpu[1] ≈ grad[1] rtol = 1.0f-5 atol = 1.0f-6
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

        empty_gpu = WeightedArray(JLArray(Float32[]), JLArray(Float32[]))
        s_empty = sprint(show, MIME"text/plain"(), empty_gpu)
        @test !isempty(s_empty)

        io_empty = IOBuffer()
        ctx_empty = IOContext(io_empty, :compact => true)
        show(ctx_empty, MIME"text/plain"(), empty_gpu)
        @test !isempty(String(take!(io_empty)))

        data_gpu_mat = WeightedArray(JLArray(Float32[1.0 2.0; 3.0 4.0]), JLArray(Float32[0.5 0.2; 1.0 0.8]))
        io = IOBuffer()
        ctx = IOContext(io, :limit => true, :displaysize => (4, 80))
        show(ctx, MIME"text/plain"(), data_gpu_mat)
        s_limited = String(take!(io))
        @test occursin("…", s_limited)
    end

end
