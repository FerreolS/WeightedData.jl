
@testset "WeightedArray.jl" begin
    A = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]


    @test WeightedData.get_value(A) == [1.0, 2.0]
    @test WeightedData.get_precision(A) == [0.5, 0.5]


    B = deepcopy(A)
    A = WeightedArray(A)
    @test B == A

    @test @inferred(flagbadpix(A, [true, false])) == [WeightedValue(0.0, 0.0), WeightedValue(2.0, 0.5)]
    @test WeightedArray([1.0, missing]) == [WeightedValue(1.0, 1.0), WeightedValue(0.0, 0.)]
    @test WeightedArray(ones(2, 3)) == WeightedArray(ones(2, 3), ones(2, 3))
    @test WeightedArray([missing, missing]) == [WeightedValue(0.0, 0.0), WeightedValue(0.0, 0.)]


    A = [1.0, 2.0, 3.0]
    B = [0.1, 0.2, 0.3]
    weighted_array = WeightedArray(A, B)
    @test get_value(weighted_array) == A
    @test get_precision(weighted_array) == B
    @test propertynames(weighted_array) == (:value, :precision)
    @test weighted_array.value == A
    @test weighted_array.precision == B


    @test size(weighted_array) == (3,)
    @test @inferred(get_value(weighted_array)) == A
    @test @inferred(get_precision(weighted_array)) == B

    badpix = [false, true, false]
    flagged_array = flagbadpix(weighted_array, badpix)
    @test get_value(flagged_array) == [1.0, 0.0, 3.0]
    @test get_precision(flagged_array) == [0.1, 0.0, 0.3]

    flagbadpix!(weighted_array, badpix)
    @test get_value(weighted_array) == [1.0, 0.0, 3.0]
    @test get_precision(weighted_array) == [0.1, 0.0, 0.3]

    C = WeightedArray([1.0, 2.0, 3.0], [1, 1, 0.5])
    @test @inferred(weightedmean(C)) == WeightedValue{Float64}(1.8, 2.5)
    @test @inferred(weightedmean(C, C)) == WeightedArray([1.0, 2.0, 3.0], [2, 2, 1])

    D = WeightedArray(ones(Float32, 2, 2), ones(2, 2))
    @test @inferred(weightedmean(D; dims=2)) == WeightedArray(ones(Float32, 2, 1), 2 * ones(2, 1))


end