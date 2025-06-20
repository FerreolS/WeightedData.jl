using WeightedData
using Test

@testset "WeightedData.jl" begin
    @testset "WeightedValue" begin
        A = WeightedValue(1.0, 0.5)
        B = WeightedValue(2.0, 0.5)

        @test A.value == 1.0
        @test A.precision == 0.5

        @test WeightedValue{Float32}(2, 1) == WeightedValue(2.0f0, 1.0f0)

        @test real(A) == WeightedValue(1.0, 0.5)
        @test imag(A) == WeightedValue(0.0, Int(0))

        @test A + B == WeightedValue(3.0, 0.25)
        @test A + 1.0 == WeightedValue(2.0, 0.5)
        @test 1.0 + A == WeightedValue(2.0, 0.5)

        @test A - B == WeightedValue(-1.0, 0.25)
        @test A - 1.0 == WeightedValue(0.0, 0.5)

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
        @test weightedmean(()) === nothing #WeightedValue(0.0, Inf)

        @test convert(Float32, A) == WeightedValue(1.0f0, 0.5f0)

        @test @inferred weightedmean(A, B) == WeightedValue(1.5, 1.0)
        @test @inferred weightedmean(A) == A
        @test @inferred weightedmean((A, B, A, B)) == WeightedValue(1.5, 2.0)
        @test @inferred weightedmean((A, B, A, B)...) == WeightedValue(1.5, 2.0)


    end
    include("WeightedArray_test.jl")
    include("likelihood_test.jl")
    include("WeightedDataPlotsExt_test.jl")
    include("WeightedDataRobustModelsExt_test.jl")
    include("WeightedDataTwoDimensionalExt_test.jl")

end
