using Zygote
@testset "likelihood.jl" begin
    A = [WeightedPoint(1.0, 1.0), WeightedPoint(2.0, 0.5)]
    B = [1.0, 1.0]

    @test @inferred gausslikelihood(A[1], B[1]) == 0.0
    @test @inferred gausslikelihood(A[2], B[2]) == 0.25

    @test @inferred robustlikelihood(3)(A[1], B[1]) == 0.0
    @test @inferred robustlikelihood(3)(A[2], B[2]) ≈ log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    @test @inferred likelihood(A, B) == 0.25

    @test @inferred likelihood(A, B, likelihoodfunc=robustlikelihood(3)) ≈ log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    C = WeightedPoint(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred scaledlikelihood(C, D) == 0.0
    @test @inferred likelihood(C, D) == 2.0

    f(x) = likelihood(C, x)
    @test @inferred Zygote.withgradient(f, D) == (data=2.0, grad=([-1.0 -1.0; -1.0 -1.0],))

    g(x) = scaledlikelihood(C, x)
    @test @inferred Zygote.withgradient(g, D) == (data=0.0, grad=([0.0 0.0; 0.0 0.0],))


    C = WeightedArray(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred scaledlikelihood(C, D) == 0.0
    @test @inferred likelihood(C, D) == 2.0

    f2(x) = likelihood(C, x)
    @test @inferred Zygote.withgradient(f2, D) == (data=2.0, grad=([-1.0 -1.0; -1.0 -1.0],))

    g2(x) = scaledlikelihood(C, x)
    @test @inferred Zygote.withgradient(g2, D) == (data=0.0, grad=([0.0 0.0; 0.0 0.0],))

end