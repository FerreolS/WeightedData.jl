using RobustModels
import WeightedData: value, precision
@testset "WeightedDataRobustModelsExt" begin
    # Test likelihood method
    @testset "loglikelihood" begin
        loss = L2Loss()
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        @test loglikelihood(loss, data, model) == RobustModels.rho(loss, sqrt(precision(data)) * (model - value(data)))

        data = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]
        model = [2.0, 3.0]
        @test loglikelihood(loss, data, model) == sum(RobustModels.rho.(loss, sqrt.(precision(data)) .* (model .- value(data))))
    end

    # Test get_weights method
    @testset "get_weights" begin
        loss = L2Loss()
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        @test get_weights(loss, data, model) == RobustModels.weight(loss, sqrt(precision(data)) * (model - value(data)))

        data = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]
        model = [2.0, 3.0]
        @test get_weights(loss, data, model) == RobustModels.weight.(loss, sqrt.(precision(data)) .* (model .- value(data)))
        @test workingweights(loss, data, model) == get_weights(loss, data, model)

        bad_model = [2.0, 3.0, 4.0]
        @test_throws ErrorException loglikelihood(loss, data, bad_model)
        @test_throws ErrorException get_weights(loss, data, bad_model)
    end

    # Test likelihood method with different loss functions
    @testset "likelihood with different loss functions" begin
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        losses = [L2Loss(), L1Loss(), HuberLoss()]
        for loss in losses
            @test loglikelihood(loss, data, model) == RobustModels.rho(loss, sqrt(precision(data)) * (model - value(data)))
        end
    end
end
