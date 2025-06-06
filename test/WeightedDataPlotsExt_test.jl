

@testsnippet Plots begin
    using Plots
end

# Test for the first recipe function
@testitem "WeightedDataPlotsExt recipe function 1" setup = [Plots] begin
    values = [1.0, 2.0, 3.0]
    precisions = [1.0, 0.5, 0.25]
    A = WeightedPoint(values, precisions)

    plot_result = plot(A)

    @test plot_result.series_list[1][:y] == values
    @test plot_result.series_list[1][:ribbon] == 3 .* sqrt.(1 ./ precisions)
    @test plot_result.series_list[1][:fillalpha] == 0.5
end

# Test for the second recipe function
@testitem "WeightedDataPlotsExt recipe function 2" setup = [Plots] begin
    x = [1, 2, 3]
    values = [1.0, 2.0, 3.0]
    precisions = [1.0, 0.5, 0.25]
    A = WeightedPoint(values, precisions)

    plot_result = plot(x, A)

    @test plot_result.series_list[1][:x] == x
    @test plot_result.series_list[1][:y] == values
    @test plot_result.series_list[1][:ribbon] == 3 .* sqrt.(1 ./ precisions)
    @test plot_result.series_list[1][:fillalpha] == 0.5
end