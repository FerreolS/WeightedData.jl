using RobustModels
@testset "WeightedDataRobustModelsExt" begin
    # Test likelihood method
    @testset "likelihood" begin
        loss = L2Loss()
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        @test likelihood(loss, data, model) == RobustModels.rho(loss, sqrt(get_precision(data)) * (model - get_value(data)))

        data = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]
        model = [2.0, 3.0]
        @test likelihood(loss, data, model) == sum(RobustModels.rho.(loss, sqrt.(get_precision(data)) .* (model .- get_value(data))))
    end

    # Test get_weight method
    @testset "get_weight" begin
        loss = L2Loss()
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        @test get_weight(loss, data, model) == RobustModels.weight(loss, sqrt(get_precision(data)) * (model - get_value(data)))

        data = [WeightedValue(1.0, 0.5), WeightedValue(2.0, 0.5)]
        model = [2.0, 3.0]
        @test get_weight(loss, data, model) == RobustModels.weight.(loss, sqrt.(get_precision(data)) .* (model .- get_value(data)))
    end

    # Test likelihood method with different loss functions
    @testset "likelihood with different loss functions" begin
        data = WeightedValue(1.0, 0.5)
        model = 2.0
        losses = [L2Loss(), L1Loss(), HuberLoss()]
        for loss in losses
            @test likelihood(loss, data, model) == RobustModels.rho(loss, sqrt(get_precision(data)) * (model - get_value(data)))
        end
    end
end