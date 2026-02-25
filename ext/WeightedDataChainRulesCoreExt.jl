module WeightedDataChainRulesCoreExt
import ChainRulesCore: NoTangent, ZeroTangent
import ChainRulesCore
import WeightedData: likelihood, L2Loss, get_value, get_precision, WeightedValue, ScaledL2Loss


"""
    ChainRulesCore.rrule(::typeof(likelihood), ::L2Loss, data, model)

Custom reverse-mode rule for `likelihood(::L2Loss, data, model)` with weighted
array data. Returns the scalar loss and a pullback that propagates gradients to
`model`.
"""
function ChainRulesCore.rrule(::typeof(likelihood), ::L2Loss, data::AbstractArray{WeightedValue{T}, N}, model::AbstractArray{T, N}) where {T, N}
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
    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp .* Δy)
    return 1 / 2 * l, likelihood_pullback
end

"""
    ChainRulesCore.rrule(::typeof(likelihood), ::ScaledL2Loss, weighteddata, model)

Custom reverse-mode rule for the scaled L2 likelihood. The pullback returns
gradients with respect to `model` leveraging the fact the gradients of  the 
scaling factor `α` with respect to the model is zero as `α` is optimal in the maximum 
likelihood sens.
"""
function ChainRulesCore.rrule(::typeof(likelihood), (; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
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
    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end

end
