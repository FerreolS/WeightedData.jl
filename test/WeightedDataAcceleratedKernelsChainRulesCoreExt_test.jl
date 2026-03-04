using ChainRulesCore
using AcceleratedKernels
import StatsAPI: loglikelihood

@testset "WeightedDataAcceleratedKernelsChainRulesCoreExt" begin
    @testset "L2Loss rrule value and pullback" begin
        values = Float32[1.0, 2.0, 3.0]
        precisions = Float32[0.5, 0.2, 1.5]
        model = Float32[2.0, 3.0, 2.5]

        weighted = WeightedArray(values, precisions)
        val, pb = ChainRulesCore.rrule(loglikelihood, WeightedData.L2Loss(), weighted, model)

        r = model .- values
        rp = precisions .* r
        ref = sum(r .* rp) / 2

        @test val ≈ ref rtol = 1f-5 atol = 1f-6
        @test pb(1f0)[4] ≈ rp rtol = 1f-5 atol = 1f-6
    end

    @testset "L2Loss shape mismatch throws" begin
        weighted = WeightedArray(Float32[1.0, 2.0, 3.0], Float32[0.5, 0.2, 1.5])
        model_bad = Float32[1.0, 2.0]

        @test_throws ErrorException("likelihood : size(A) != size(model)") ChainRulesCore.rrule(loglikelihood, WeightedData.L2Loss(), weighted, model_bad)
    end


end
