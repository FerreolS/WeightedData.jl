using WeightedData
using Test
using Coverage

# Start coverage analysis
Coverage.start()

@testset "WeightedData.jl" begin
    @testset "WeightedPoint" begin
        A = WeightedPoint(1.0, 0.5)
        B = WeightedPoint(2.0, 0.5)

        @test A.val == 1.0
        @test A.precision == 0.5

        @test real(A) == WeightedPoint(1.0, 0.5)
        @test imag(A) == WeightedPoint(0.0, 0.0)

        @test A + B == WeightedPoint(3.0, 0.25)
        @test A + 1.0 == WeightedPoint(2.0, 0.5)
        @test 1.0 + A == WeightedPoint(2.0, 0.5)

        @test A - B == WeightedPoint(-1.0, 0.25)
        @test A - 1.0 == WeightedPoint(0.0, 0.5)

        @test A / 2.0 == WeightedPoint(0.5, 2.0)
        @test 2.0 * A == WeightedPoint(2.0, 0.125)
        @test A * 2.0 == WeightedPoint(2.0, 0.125)

        @test combine(A, B) == WeightedPoint(1.5, 1.0)
    end
    @testset "arrays.jl" begin
        A = [WeightedPoint(1.0, 0.5), WeightedPoint(2.0, 0.5)]
        B = [WeightedPoint(2.0, 0.5), WeightedPoint(3.0, 0.5)]

        @test WeightedData.get_val(A) == [1.0, 2.0]
        @test WeightedData.get_precision(A) == [0.5, 0.5]

        @test flagbadpix(A, [true, false]) == [WeightedPoint(0.0, 0.0), WeightedPoint(2.0, 0.5)]
    end
    @testset "likelihood.jl" begin
        A = [WeightedPoint(1.0, 1.0), WeightedPoint(2.0, 0.5)]
        B = [1.0, 1.0]

        @test gausslikelihood(A[1], B[1]) == 0.0
        @test gausslikelihood(A[2], B[2]) == 0.25

        @test robustlikelihood(3)(A[1], B[1]) == 0.0

        @test likelihood(A, B) == 0.25

    end
end

# Stop coverage analysis and save results
Coverage.stop()
Coverage.report()
