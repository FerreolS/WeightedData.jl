@testset "likelihood.jl" begin
    using DifferentiationInterface, Zygote, ForwardDiff, ChainRulesCore
    import StatsAPI: loglikelihood
    A = [WeightedValue(1.0, 1.0), WeightedValue(2.0, 0.5)]
    B = [1.0, 1.0]

    @test @inferred(loglikelihood(A[1], B[1])) == 0.0
    @test @inferred(loglikelihood(A[2], B[2])) ≈ 0.25

    #@test @inferred cauchyloss(3)(A[1], B[1]) == 0.0
    #@test @inferred cauchyloss(3)(A[2], B[2]) ≈ (3 / 2.385)^(-2) / 2 * log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    @test @inferred(loglikelihood(A, B)) == 0.25
    @test_deprecated WeightedData.likelihood(A, B)

    #@test @inferred likelihood(A, B, loss=cauchyloss(3)) ≈ (3 / 2.385)^(-2) / 2 * log(1 + ((3 / 2.385) * sqrt(0.5) * (1 - 2))^2)

    C = WeightedValue(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred(loglikelihood(C, D; loss = ScaledL2Loss())) == 0.0
    @test @inferred(loglikelihood(C, D)) == 2.0

    f(x) = loglikelihood(C, x)
    @test Zygote.withgradient(f, D) == (val = 2.0, grad = ([-1.0 -1.0; -1.0 -1.0],))

    g(x) = loglikelihood(C, x, loss = ScaledL2Loss())
    @test Zygote.withgradient(g, D) == (val = 0.0, grad = ([0.0 0.0; 0.0 0.0],))
    @test WeightedData.get_weights(C, D) == ones(2, 2)

    @test WeightedData.get_weights(A[1], B[1]) == get_precision(A[1])

    bad_model = ones(3, 2)
    @test_throws ErrorException loglikelihood(C, bad_model)
    @test_throws ErrorException WeightedData.get_weights(C, bad_model)


    C = WeightedArray(1.0 .+ ones(2, 2), ones(2, 2))
    D = ones(2, 2)

    @test @inferred(loglikelihood(C, D, loss = ScaledL2Loss())) == 0.0
    @test @inferred(loglikelihood(C, D)) == 2.0

    f2(x) = loglikelihood(C, x)
    @test Zygote.withgradient(f2, D) == (val = 2.0, grad = ([-1.0 -1.0; -1.0 -1.0],))

    grad = similar(D)
    @test @inferred(DifferentiationInterface.value_and_gradient!(f, grad, AutoForwardDiff(), D)) == (2.0, [-1.0 -1.0; -1.0 -1.0])

    g2(x) = loglikelihood(C, x, loss = ScaledL2Loss(nonnegative = true))
    @test Zygote.withgradient(g2, D) == (val = 0.0, grad = ([0.0 0.0; 0.0 0.0],))

    data_nonneg = WeightedArray([1.0, 2.0], [1.0, 1.0])
    model_neg = [-1.0, -2.0]
    @test loglikelihood(data_nonneg, model_neg, loss = ScaledL2Loss(nonnegative = true)) == 2.5

    model_zero = [0.0, 0.0]
    @test loglikelihood(data_nonneg, model_zero, loss = ScaledL2Loss(nonnegative = true)) == 2.5

    weighted_nanalpha = WeightedArray([1.0, 2.0], [1.0, 1.0])
    model_nanalpha = [0.0, 0.0]
    val_rrule_nanalpha, pb_rrule_nanalpha = ChainRulesCore.rrule(loglikelihood, ScaledL2Loss(), weighted_nanalpha, model_nanalpha)
    @test val_rrule_nanalpha == 2.5
    @test pb_rrule_nanalpha(1.0)[4] == [0.0, 0.0]

    weighted_negativealpha = WeightedArray([1.0, 2.0], [1.0, 1.0])
    model_negativealpha = [-1.0, -2.0]
    val_rrule_negativealpha, pb_rrule_negativealpha = ChainRulesCore.rrule(loglikelihood, ScaledL2Loss(nonnegative = true), weighted_negativealpha, model_negativealpha)
    @test val_rrule_negativealpha == 2.5
    @test pb_rrule_negativealpha(1.0)[4] == [0.0, 0.0]

    bad_model_rrule = ones(3, 2)
    @test_throws ErrorException ChainRulesCore.rrule(loglikelihood, WeightedData.L2Loss(), C, bad_model_rrule)
    @test_throws ErrorException ChainRulesCore.rrule(loglikelihood, ScaledL2Loss(), C, bad_model_rrule)

end
