
@testset "WeightedArray.jl" begin
    A = [WeightedPoint(1.0, 0.5), WeightedPoint(2.0, 0.5)]
    B = [WeightedPoint(2.0, 0.5), WeightedPoint(3.0, 0.5)]

    @test WeightedData.get_val(A) == [1.0, 2.0]
    @test WeightedData.get_precision(A) == [0.5, 0.5]

    @test @inferred flagbadpix(A, [true, false]) == [WeightedPoint(0.0, 0.0), WeightedPoint(2.0, 0.5)]

    A = [1.0, 2.0, 3.0]
    B = [0.1, 0.2, 0.3]
    weighted_array = WeightedArray(A, B)

    @test size(weighted_array) == (3,)
    @test @inferred get_val(weighted_array) == A
    @test @inferred get_precision(weighted_array) == B

    badpix = [false, true, false]
    flagged_array = flagbadpix(weighted_array, badpix)
    @test get_val(flagged_array) == [1.0, 0.0, 3.0]
    @test get_precision(flagged_array) == [0.1, 0.0, 0.3]

    flagbadpix!(weighted_array, badpix)
    @test get_val(weighted_array) == [1.0, 0.0, 3.0]
    @test get_precision(weighted_array) == [0.1, 0.0, 0.3]

    C = WeightedArray([1.0, 2.0, 3.0], [1, 1, 0.5])
    @test @inferred combine(C) == WeightedPoint{Float64}(1.8, 2.5)
    @test @inferred combine(C, C) == WeightedArray([1.0, 2.0, 3.0], [2, 2, 1])

    D = WeightedArray(ones(Float32, 2, 2), ones(2, 2))
    @test @inferred combine(D; dims=2) == WeightedArray(ones(Float32, 2, 1), 2 * ones(2, 1))


end