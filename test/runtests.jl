using WeightedData
using Test
import TypeUtils

@test !isdefined(@__MODULE__, :weightedmean)
@test !isdefined(@__MODULE__, :ScaledL2Loss)

import WeightedData: weightedmean, ScaledL2Loss, flagbaddata, flagbaddata!

@testset "WeightedData.jl" begin
    @testset "WeightedValue" begin
        A = WeightedValue(1.0, 0.5)
        B = WeightedValue(2.0, 0.5)

        @test A.value == 1.0
        @test A.precision == 0.5

        @test WeightedValue{Float32}(2, 1) == WeightedValue(2.0f0, 1.0f0)
        @test TypeUtils.get_precision(A) == Float64
        @test TypeUtils.get_precision(WeightedValue{Float32}(2, 1)) == Float32

        @test real(A) == WeightedValue(1.0, 0.5)
        @test imag(A) == WeightedValue(0.0, Int(0))

        @test A + B == WeightedValue(3.0, 0.25)
        @test A + 1.0 == WeightedValue(2.0, 0.5)
        @test 1.0 + A == WeightedValue(2.0, 0.5)

        @test A - B == WeightedValue(-1.0, 0.25)
        @test A - 1.0 == WeightedValue(0.0, 0.5)
        @test_throws ErrorException WeightedValue{Float64}(1.0, -1.0)

        @test 2.0 * A == WeightedValue(2.0, 0.125)
        @test A * 2.0 == WeightedValue(2.0, 0.125)
        # Test for multiplication error
        @test_throws ErrorException WeightedValue(1.0, 0.5) * WeightedValue(2.0, 1.5)

        @test A / 2.0 == WeightedValue(0.5, 2.0)
        @test 2.0 / A == B
        @test_throws ErrorException WeightedValue(1.0, 0.5) / WeightedValue(2.0, 1.5)

        # Test for Base.one
        @test one(A) * A == A

        # Test for Base.zero
        @test zero(A) + A == A
        @test_throws ArgumentError weightedmean(())

        #@test convert(Float32, A) == WeightedValue(1.0f0, 0.5f0)

        @test @inferred weightedmean(A, B) == WeightedValue(1.5, 1.0)
        @test @inferred weightedmean(A) == A
        @test @inferred weightedmean((A, B, A, B)) == WeightedValue(1.5, 2.0)
        @test @inferred weightedmean((A, B, A, B)...) == WeightedValue(1.5, 2.0)


    end
    @testset "show method for WeightedValue" begin
        wv1 = WeightedValue(1.0, 4.0)
        s1 = sprint(show, wv1)
        @test s1 == "1.0 ± 0.5"

        wv2 = WeightedValue(2.5, 0.25)
        s2 = sprint(show, wv2)
        @test s2 == "2.5 ± 2.0"

        wv3 = WeightedValue(10.0, 100.0)
        s3 = sprint(show, wv3)
        @test s3 == "10.0 ± 0.1"

        # Test with infinite precision
        wv4 = WeightedValue(5.0, Inf)
        s4 = sprint(show, wv4)
        @test s4 == "5.0 ± 0.0"

        # Test with zero precision
        wv5 = WeightedValue(7.0, 0.0)
        s5 = sprint(show, wv5)
        @test s5 == "7.0 ± Inf"
    end

    include("WeightedArray_test.jl")
    include("likelihood_test.jl")
    include("WeightedDataPlotsExt_test.jl")
    include("WeightedDataRobustModelsExt_test.jl")
    include("WeightedDataTwoDimensionalExt_test.jl")
    include("WeightedDataMeasurementsExt_test.jl")
    include("WeightedDataOnlineSampleStatisticsExt_test.jl")
    include("WeightedDataUncertainExt_test.jl")

end
