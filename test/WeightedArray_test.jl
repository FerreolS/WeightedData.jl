@testset "WeightedArray.jl" begin
    A = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]


    @test WeightedData.get_value(A) == [1.0, 2.0]
    @test WeightedData.get_precision(A) == [0.5, 0.5]


    B = deepcopy(A)
    @test B == WeightedArray(A)
    @test B == WeightedArray(WeightedArray(A))

    @test @inferred(flagbadpix(A, [true, false])) == [WeightedValue(0.0, 0.0), WeightedValue(2.0, 0.5)]
    @test WeightedArray([1.0, missing]) == [WeightedValue(1.0, 1.0), WeightedValue(0.0, 0.0)]
    @test WeightedArray(ones(2, 3)) == WeightedArray(ones(2, 3), ones(2, 3))
    @test WeightedArray([missing, missing]) == [WeightedValue(0.0, 0.0), WeightedValue(0.0, 0.0)]


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

    @test reshape(weighted_array, 3, 1) == WeightedArray(reshape(A, 3, 1), reshape(B, 3, 1))
    @test reshape(weighted_array, :) == WeightedArray(reshape(A, :), reshape(B, :))

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
    @test @inferred(weightedmean(D; dims = 2)) == WeightedArray(ones(Float32, 2, 1), 2 * ones(2, 1))


end

@testset "WeightedArray arithmetic operations" begin
    A = WeightedArray([1.0, 2.0], [0.5, 0.2])
    B = WeightedArray([3.0, 4.0], [0.5, 0.8])

    # Test addition
    C = A + B
    @test C isa WeightedArray
    @test get_value(C) ≈ [4.0, 6.0]
    @test get_precision(C) ≈ [0.25, 0.16] # inv(inv(0.5)+inv(0.5)) = 0.25, inv(inv(0.2)+inv(0.8)) = 0.16
    @test C == A .+ B

    # Test subtraction
    D = A - B
    @test D isa WeightedArray
    @test D == A .- B
    @test get_value(D) ≈ [-2.0, -2.0]
    @test get_precision(D) ≈ [0.25, 0.16]

    # Test WeightedArray + Number
    E = A + 2.0
    @test E isa WeightedArray
    @test E == A .+ 2.0
    @test get_value(E) ≈ [3.0, 4.0]
    @test get_precision(E) == [0.5, 0.2]

    # Test Number + WeightedArray
    F = 2.0 + A
    @test F isa WeightedArray
    @test F == 2.0 .+ A
    @test get_value(F) ≈ [3.0, 4.0]
    @test get_precision(F) == [0.5, 0.2]

    # Test WeightedArray - Number
    G = A - 1.0
    @test G isa WeightedArray
    @test G == A .- 1.0
    @test get_value(G) ≈ [0.0, 1.0]
    @test get_precision(G) == [0.5, 0.2]

    # Test Number - WeightedArray
    H = 5.0 - A
    @test H isa WeightedArray
    @test H == 5.0 .- A
    @test get_value(H) ≈ [4.0, 3.0]
    @test get_precision(H) == [0.5, 0.2]

    # Test WeightedArray / Number
    I = A / 2.0
    @test I isa WeightedArray
    @test I == A ./ 2.0
    @test get_value(I) ≈ [0.5, 1.0]
    @test get_precision(I) ≈ [2.0, 0.8] # 2^2 * precision

    # Test Number / WeightedArray
    J = 2.0 / A
    @test J isa WeightedArray
    @test J == 2.0 ./ A
    @test get_value(J) ≈ [2.0, 1.0]
    @test get_precision(J) ≈ [1.0 / (0.5 * 4.0), 1.0 / (0.2 * 4.0)] # inv(precision) / 2^2

    # Test Number * WeightedArray
    K = 3.0 * A
    @test K isa WeightedArray
    @test K == 3.0 .* A
    @test get_value(K) ≈ [3.0, 6.0]
    @test get_precision(K) ≈ [0.5 / 9.0, 0.2 / 9.0] # precision / 3^2

    # Test WeightedArray * Number
    L = A * 3.0
    @test L isa WeightedArray
    @test L == A .* 3.0
    @test get_value(L) ≈ [3.0, 6.0]
    @test get_precision(L) ≈ [0.5 / 9.0, 0.2 / 9.0]

    # Test unsupported WeightedArray * WeightedArray
    @test_throws ErrorException A * B

    # Test unsupported WeightedArray / WeightedArray
    @test_throws ErrorException A / B
end

@testset "WeightedArray arithmetic with   arrays" begin
    A = WeightedArray([1.0, 2.0], [0.5, 0.2])


    # WeightedArray + Array
    arr = [2.0, 3.0]
    res = A + arr
    @test get_value(res) == [3.0, 5.0]
    @test get_precision(res) == [0.5, 0.2]


    # Array + WeightedArray
    @test get_value(arr + A) == [3.0, 5.0]
    @test get_precision(arr + A) == [0.5, 0.2]
    @test (arr + A) == arr .+ A

    # WeightedArray - Array
    @test get_value(A - arr) == [-1.0, -1.0]
    @test get_precision(A - arr) == [0.5, 0.2]
    @test (A - arr) == A .- arr

    # Array - WeightedArray
    @test get_value(arr - A) == [1.0, 1.0]
    @test get_precision(arr - A) == [0.5, 0.2]
    @test (arr - A) == arr .- A

    # WeightedArray / Array
    arr2 = [2.0, 4.0]
    res2 = A ./ arr2
    @test get_value(res2) == [0.5, 0.5]
    @test get_precision(res2) ≈ [2.0, 3.2]

    # Array / WeightedArray
    arr3 = [2.0, 4.0]
    res3 = arr3 ./ A
    @test get_value(res3) == [2.0, 2.0]
    @test get_precision(res3) ≈ [0.5, 0.3125]

    # Array * WeightedArray
    arr4 = [2.0, 4.0]
    res4 = arr4 .* A
    @test get_value(res4) == [2.0, 8.0]
    @test get_precision(res4) ≈ [0.125, 0.0125]

    # WeightedArray * Array
    arr5 = [2.0, 4.0]
    res5 = A .* arr5
    @test get_value(res5) == [2.0, 8.0]
    @test get_precision(res5) ≈ [0.125, 0.0125]
end
