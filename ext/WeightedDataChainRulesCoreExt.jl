module WeightedDataChainRulesCoreExt
import ChainRulesCore: NoTangent, ZeroTangent
import ChainRulesCore
import StatsAPI: loglikelihood
import WeightedData: L2Loss, get_value, get_precision, WeightedValue, ScaledL2Loss


"""
    ChainRulesCore.rrule(::typeof(loglikelihood), ::L2Loss, data, model)

Custom reverse-mode rule for
`loglikelihood(::L2Loss, data::AbstractArray{<:WeightedValue}, model)`.

# Arguments
- `data`: weighted observations
- `model`: model values with the same shape as `data`

# Returns
- scalar objective value
- pullback that propagates gradients to `model`
"""
function ChainRulesCore.rrule(::typeof(loglikelihood), ::L2Loss, data::AbstractArray{WeightedValue{T}, N}, model::AbstractArray{T, N}) where {T, N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")

    d = get_value(data)
    p = get_precision(data)
    rp = similar(d)
    l = T(0)
    @inbounds @simd for i in eachindex(data, model)
        r = model[i] - d[i]
        rp[i] = p[i] * r
        l += r .* rp[i]
    end
    loglikelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp .* Δy)
    return 1 / 2 * l, loglikelihood_pullback
end

"""
    ChainRulesCore.rrule(::typeof(loglikelihood), ::ScaledL2Loss, weighteddata, model)

Custom reverse-mode rule for
`loglikelihood(::ScaledL2Loss, weighteddata::AbstractArray{<:WeightedValue}, model)`.

The pullback returns gradients with respect to `model`. The derivative through
the optimal scaling factor `α` is intentionally ignored (treated as stationary
at optimum).

# Arguments
- `weighteddata`: weighted observations
- `model`: model values with the same shape as `weighteddata`

# Returns
- scalar objective value
- pullback that propagates gradients to `model`
"""
function ChainRulesCore.rrule(::typeof(loglikelihood), (; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(weighteddata) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_value(weighteddata)
    precision = get_precision(weighteddata)


    a = similar(data)
    b = similar(data)
    @inbounds @simd for i in eachindex(data, model)
        b[i] = model[i] .* precision[i] .* data[i]
        a[i] = model[i] .* precision[i] .* model[i]
    end

    α = sum(b, dims = dims) ./ sum(a, dims = dims)

    @inbounds @simd for i in eachindex(α)
        if (nonnegative && α[i] < 0) || !isfinite(α[i])
            α[i] = T2(0)
        end
    end
    r = (α .* model .- data)
    rp = r .* precision
    loglikelihood_pullback(Δy) = (NoTangent(), NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, loglikelihood_pullback
end

end
