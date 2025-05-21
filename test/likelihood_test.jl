using DifferentiationInterface, Zygote, ForwardDiff
@testset "likelihood.jl" begin
    A = [WeightedPoint(1.0, 1.0), WeightedPoint(2.0, 0.5)]
    B = [1.0, 1.0]

    @test @inferred(likelihood(A[1], B[1])) == 0.0
    @test @inferred(likelihood(A[2], B[2])) ≈ 0.25

    #@test @inferred cauchyloss(3)(A[1], B[1]) == 0.0
    #@test @inferred cauchyloss(3)(A[2], B[2]) ≈ (3 / 2.385)^(-2) / 2 * log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    @test @inferred(likelihood(A, B)) == 0.25

    #@test @inferred likelihood(A, B, loss=cauchyloss(3)) ≈ (3 / 2.385)^(-2) / 2 * log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    C = WeightedPoint(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred(likelihood(C, D; loss=ScaledL2Loss())) == 0.0
    @test @inferred(likelihood(C, D)) == 2.0

    f(x) = likelihood(C, x)
    @test Zygote.withgradient(f, D) == (val=2.0, grad=([-1.0 -1.0; -1.0 -1.0],))

    g(x) = likelihood(C, x, loss=ScaledL2Loss())
    @test Zygote.withgradient(g, D) == (val=0.0, grad=([0.0 0.0; 0.0 0.0],))


    C = WeightedArray(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred(likelihood(C, D, loss=ScaledL2Loss())) == 0.0
    @test @inferred(likelihood(C, D)) == 2.0

    f2(x) = likelihood(C, x)
    @test Zygote.withgradient(f2, D) == (val=2.0, grad=([-1.0 -1.0; -1.0 -1.0],))

    grad = similar(D)
    @test @inferred(DifferentiationInterface.value_and_gradient!(f, grad, AutoForwardDiff(), D)) == (2.0, [-1.0 -1.0; -1.0 -1.0])

    g2(x) = likelihood(C, x, loss=ScaledL2Loss(nonnegative=true))
    @test Zygote.withgradient(g2, D) == (val=0.0, grad=([0.0 0.0; 0.0 0.0],))

end