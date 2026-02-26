using OnlineSampleStatistics
using WeightedData
import WeightedData: value, precision

@testset "WeightedDataOnlineSampleStatisticsExt" begin
    @testset "WeightedValue from UnivariateStatistic" begin
        s = UnivariateStatistic([1.0, 2.0, 3.0], 2)
        w = WeightedValue(s)

        @test value(w) ≈ mean(s)
        @test precision(w) ≈ inv(var(s))
    end

    @testset "WeightedValue requires variance moment" begin
        s = UnivariateStatistic([1.0, 2.0, 3.0], 1)
        @test_throws ArgumentError WeightedValue(s)
    end

    @testset "WeightedArray from IndependentStatistic" begin
        x = cat(
            [1.0 2.0; 3.0 4.0],
            [2.0 1.0; 4.0 3.0],
            [3.0 0.0; 5.0 2.0]; dims = 3
        )
        s = IndependentStatistic(x, 2; dims = 3)
        w = WeightedArray(s)

        @test value(w) ≈ mean(s)
        @test precision(w) ≈ inv.(var(s))
    end

    @testset "WeightedArray requires variance moment" begin
        x = cat([1.0 2.0], [2.0 1.0], [3.0 0.0]; dims = 2)
        s = IndependentStatistic(x, 1; dims = 2)
        @test_throws ArgumentError WeightedArray(s)
    end
end
