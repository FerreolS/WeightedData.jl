using WeightedData
using Test

@testset "WeightedData.jl" begin
    @testset "WeightedPoint" begin
        A = WeightedPoint(1.0, 0.5)
        B = WeightedPoint(2.0, 0.5)

        @test A.val == 1.0
        @test A.precision == 0.5

        @test real(A) == WeightedPoint(1.0, 0.5)
        @test imag(A) == WeightedPoint(0.0, Int(0))

        @test A + B == WeightedPoint(3.0, 0.25)
        @test A + 1.0 == WeightedPoint(2.0, 0.5)
        @test 1.0 + A == WeightedPoint(2.0, 0.5)

        @test A - B == WeightedPoint(-1.0, 0.25)
        @test A - 1.0 == WeightedPoint(0.0, 0.5)

        @test 2.0 * A == WeightedPoint(2.0, 0.125)
        @test A * 2.0 == WeightedPoint(2.0, 0.125)
        # Test for multiplication error
        @test_throws ErrorException WeightedPoint(1.0, 0.5) * WeightedPoint(2.0, 1.5)

        @test A / 2.0 == WeightedPoint(0.5, 2.0)
        @test 2.0 / A == B
        @test_throws ErrorException WeightedPoint(1.0, 0.5) / WeightedPoint(2.0, 1.5)

        # Test for Base.one
        @test one(A) * A == A

        # Test for Base.zero
        @test zero(A) + A == A

        @test convert(Float32, A) == WeightedPoint(1.0f0, 0.5f0)

        @test @inferred combine(A, B) == WeightedPoint(1.5, 1.0)
        @test @inferred combine(A) == A
        @test @inferred combine((A, B, A, B)) == WeightedPoint(1.5, 2.0)
        @test @inferred combine([A, B, A, B]) == WeightedPoint(1.5, 2.0)
        @test @inferred combine(A, [B, A, B]) == WeightedPoint(1.5, 2.0)

    end
    @testset "arrays.jl" begin
        A = [WeightedPoint(1.0, 0.5), WeightedPoint(2.0, 0.5)]
        B = [WeightedPoint(2.0, 0.5), WeightedPoint(3.0, 0.5)]

        @test WeightedData.get_val(A) == [1.0, 2.0]
        @test WeightedData.get_precision(A) == [0.5, 0.5]

        @test @inferred flagbadpix(A, [true, false]) == [WeightedPoint(0.0, 0.0), WeightedPoint(2.0, 0.5)]
    end
    include("likelihood_test.jl")
    include("WeightedDataPlotsExt_test.jl")

end
