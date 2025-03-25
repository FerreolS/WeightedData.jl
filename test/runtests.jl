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

        @test A / 2.0 == WeightedPoint(0.5, 2.0)
        @test 2.0 * A == WeightedPoint(2.0, 0.125)
        @test A * 2.0 == WeightedPoint(2.0, 0.125)

        @test convert(Float32, A) == WeightedPoint(1.0f0, 0.5f0)

        @test combine(A, B) == WeightedPoint(1.5, 1.0)
        @test combine(A) == A
        @test combine((A, B, A, B)) == WeightedPoint(1.5, 2.0)
        @test combine([A, B, A, B]) == WeightedPoint(1.5, 2.0)
        @test combine(A, [B, A, B]) == WeightedPoint(1.5, 2.0)
    end
    @testset "arrays.jl" begin
        A = [WeightedPoint(1.0, 0.5), WeightedPoint(2.0, 0.5)]
        B = [WeightedPoint(2.0, 0.5), WeightedPoint(3.0, 0.5)]

        @test WeightedData.get_val(A) == [1.0, 2.0]
        @test WeightedData.get_precision(A) == [0.5, 0.5]

        @test flagbadpix(A, [true, false]) == [WeightedPoint(0.0, 0.0), WeightedPoint(2.0, 0.5)]
    end
    include("likelihood_test.jl")

end
