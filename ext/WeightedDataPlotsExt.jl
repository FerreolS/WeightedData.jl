module WeightedDataPlotsExt
using Plots
import WeightedData: get_data, get_precision, WeightedArray

@recipe function f(A::WeightedArray)
    data = get_data(A)
    precision = get_precision(A)
    extval = extrema(data)
    σ = sqrt.(1 ./ precision)
    σ[σ.==Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    data
end


@recipe function f(x::AbstractVector, A::WeightedArray)
    data = get_data(A)
    precision = get_precision(A)
    extval = extrema(data)
    σ = sqrt.(1 ./ precision)
    σ[σ.==Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    (x, data)
end
end