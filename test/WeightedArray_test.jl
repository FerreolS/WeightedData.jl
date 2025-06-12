
@testset "WeightedArray.jl" begin
    A = [1.0 ± 2, 2.0 ± 2.0]


    @test WeightedData.get_data(A) == [1.0, 2.0]
    @test WeightedData.get_precision(A) == [0.25, 0.25]


    B = deepcopy(A)
    A = WeightedArray(A)
    @test B == A

    @test @inferred(flagbadpix(A, [true, false])) == [0.0 ± Inf, 2.0 ± 2.0]
    @test @inferred(flagbadpix(A, [true, false])) == [0.0 ± Inf, 2.0 ± 2.0]
    @test WeightedArray([1.0, missing]) == [1.0 ± 1.0, 0.0 ± Inf]
    @test WeightedArray(ones(2, 3)) == WeightedArray(ones(2, 3), ones(2, 3))
    @test WeightedArray([missing, missing]) == [0.0 ± Inf, 0.0 ± Inf]


    A = [1.0, 2.0, 3.0]
    B = [0.1, 0.2, 0.3]
    weighted_array = WeightedArray(A, B)
    @test get_data(weighted_array) == A
    @test get_precision(weighted_array) == B
    @test propertynames(weighted_array) == (:data, :precision)
    @test weighted_array.data == A
    @test weighted_array.precision == B


    @test size(weighted_array) == (3,)
    @test @inferred(get_data(weighted_array)) == A
    @test @inferred(get_precision(weighted_array)) == B

    badpix = [false, true, false]
    flagged_array = flagbadpix(weighted_array, badpix)
    @test get_data(flagged_array) == [1.0, 0.0, 3.0]
    @test get_precision(flagged_array) ≈ [0.1, 0.0, 0.3]

    flagbadpix!(weighted_array, badpix)
    @test get_data(weighted_array) == [1.0, 0.0, 3.0]
    @test get_precision(weighted_array) == [0.1, 0.0, 0.3]

    C = WeightedArray([1.0, 2.0, 3.0], [1, 1, 0.5])
    @test @inferred(weightedmean(C)) == 1.8 ± sqrt(inv(2.5))
    @test @inferred(weightedmean((C, C))) == WeightedArray([1.0, 2.0, 3.0], [2, 2, 1])

    D = WeightedArray(ones(Float32, 2, 2), ones(2, 2))
    # @test @inferred(weightedmean(D; dims=2)) == WeightedArray(ones(Float32, 2, 1), 2 * ones(2, 1))


end