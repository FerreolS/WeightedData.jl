using RobustModels
@testset "WeightedDataRobustModelsExt" begin
    # Test likelihood method
    @testset "likelihood" begin
        loss = L2Loss()
        data = 1.0 ± √2
        model = 2.0
        @test likelihood(loss, data, model) == RobustModels.rho(loss, sqrt(get_precision(data)) * (model - get_data(data)))

        data = measurement.([1.0, 2.0], [√2, √2])
        model = [2.0, 3.0]
        @test likelihood(loss, data, model) == sum(RobustModels.rho.(loss, sqrt.(get_precision(data)) .* (model .- get_data(data))))
    end

    # Test get_weight method
    @testset "get_weight" begin
        loss = L2Loss()
        data = 1.0 ± √2
        model = 2.0
        @test get_weight(loss, data, model) == RobustModels.weight(loss, sqrt(get_precision(data)) * (model - get_data(data)))

        data = measurement.([1.0, 2.0], [√2, √2])
        model = [2.0, 3.0]
        @test get_weight(loss, data, model) == RobustModels.weight.(loss, sqrt.(get_precision(data)) .* (model .- get_data(data)))
    end

    # Test likelihood method with different loss functions
    @testset "likelihood with different loss functions" begin
        data = 1.0 ± √2
        model = 2.0
        losses = [L2Loss(), L1Loss(), HuberLoss()]
        for loss in losses
            @test likelihood(loss, data, model) == RobustModels.rho(loss, sqrt(get_precision(data)) * (model - get_data(data)))
        end
    end
end