using TwoDimensional
import WeightedData: value, precision
@testset "WeightedDataTwoDimensionalExt" begin
    # Test Base.view method
    @testset "Base.view" begin
        A = WeightedArray(rand(10, 10), rand(10, 10))
        I = BoundingBox(1:5, 1:5)
        view_A = view(A, I)
        @test size(view_A) == (5, 5)
        @test value(view_A) == view(value(A), I)
        @test precision(view_A) == view(precision(A), I)
    end
end
