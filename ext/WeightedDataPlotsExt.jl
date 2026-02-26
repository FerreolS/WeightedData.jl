module WeightedDataPlotsExt
import Plots: @recipe
import WeightedData: WeightedValue, value, precision, WeightedArray

@recipe function f(A::AbstractArray{WeightedValue{T}, N}) where {T, N}
    data = value(A)
    p = precision(A)
    extval = extrema(data)
    σ = 3 .* sqrt.(1 ./ p)
    σ[σ .== Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    data
end


@recipe function f(x, A::AbstractArray{WeightedValue{T}, N}) where {T, N}
    data = value(A)
    p = precision(A)
    extval = extrema(data)
    σ = 3 .* sqrt.(1 ./ p)
    σ[σ .== Inf] .= extval[2] - extval[1]
    ribbon := σ
    fillalpha := 0.5
    ylims := extval
    (x, data)
end
end
