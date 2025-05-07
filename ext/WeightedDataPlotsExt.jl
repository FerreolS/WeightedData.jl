module WeightedDataPlotsExt
using Plots
import WeightedData: WeightedPoint, get_val, get_precision, WeightedArray

@recipe function f(A::AbstractArray{WeightedPoint{T},N}) where {T,N}
    data = get_val(A)
    precision = get_precision(A)
    extval = extrema(data)
    σ = 3 .* sqrt.(1 ./ precision)
    σ[σ.==Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    data
end


@recipe function f(x, A::AbstractArray{WeightedPoint{T},N}) where {T,N}
    data = get_val(A)
    precision = get_precision(A)
    extval = extrema(data)
    σ = 3 .* sqrt.(1 ./ precision)
    σ[σ.==Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    (x, data)
end
end