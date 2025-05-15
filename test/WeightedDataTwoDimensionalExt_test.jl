using TwoDimensional
@testset "WeightedDataTwoDimensionalExt" begin
    # Test Base.view method
    @testset "Base.view" begin
        A = WeightedArray(rand(10, 10), rand(10, 10))
        I = BoundingBox(1:5, 1:5)
        view_A = view(A, I)
        @test size(view_A) == (5, 5)
        @test get_data(view_A) == view(get_data(A), I)
        @test get_precision(view_A) == view(get_precision(A), I)
    end
end