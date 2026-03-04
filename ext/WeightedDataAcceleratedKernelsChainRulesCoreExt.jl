module WeightedDataAcceleratedKernelsChainRulesCoreExt
__precompile__(false)

import ChainRulesCore: NoTangent, ZeroTangent
import ChainRulesCore
import StatsAPI: loglikelihood
import WeightedData: L2Loss, get_value, get_precision, WeightedValue, ScaledL2Loss
import AcceleratedKernels as AK

"""
    ChainRulesCore.rrule(::typeof(loglikelihood), ::L2Loss, data, model)

Custom reverse-mode rule for
`loglikelihood(::L2Loss, data::AbstractArray{WeightedValue{T},N}, model::AbstractArray{T,N})`
when the `WeightedDataAcceleratedKernelsChainRulesCoreExt` extension is active.

# Arguments
- `data`: weighted observations
- `model`: model values with the same shape and element type as `data`

# Constraints
- `size(data) == size(model)`
- `AK.get_backend(get_value(data)) == AK.get_backend(model)`

An `ErrorException` is thrown when one of these constraints is not satisfied.

# Returns
- scalar objective value `sum(get_precision(data) .* (model .- get_value(data)).^2) / 2`
- pullback that propagates gradients to `model` as
    `(NoTangent(), NoTangent(), NoTangent(), get_precision(data) .* (model .- get_value(data)) .* Δy)`
"""
function ChainRulesCore.rrule(::typeof(loglikelihood), ::L2Loss, data::AbstractArray{WeightedValue{T}, N}, model::AbstractArray{T, N}) where {T, N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    d = get_value(data)
    
    AK.get_backend(d) == AK.get_backend(model) || error("likelihood : data and model must be on the same backend")

    p = get_precision(data)
    rp = similar(d)
    l = AK.mapreduce(+, eachindex(model), AK.get_backend(model); init = zero(T)) do i
        r = model[i] - d[i]
        rp[i] = p[i] * r
        return r .* rp[i]
    end 
    loglikelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp .* Δy)
    return 1 / 2 * l, loglikelihood_pullback
end

"""
    ChainRulesCore.rrule(::typeof(loglikelihood), ::ScaledL2Loss, weighteddata, model)

Custom reverse-mode rule for
`loglikelihood(::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T},N}, model::AbstractArray{T,N})`
when the `WeightedDataAcceleratedKernelsChainRulesCoreExt` extension is active.

The pullback returns gradients with respect to `model`. The derivative through
the optimal scaling factor `α` is intentionally ignored (treated as stationary
at optimum).

# Arguments
- `weighteddata`: weighted observations
- `model`: model values with the same shape and element type as `weighteddata`

# Constraints
- `size(weighteddata) == size(model)`
- `AK.get_backend(get_value(weighteddata)) == AK.get_backend(model)`

An `ErrorException` is thrown when one of these constraints is not satisfied.

# Returns
- scalar objective value
- pullback that propagates gradients to `model` as
    `(NoTangent(), NoTangent(), ZeroTangent(), α .* (α .* model .- get_value(weighteddata)) .* get_precision(weighteddata) .* Δy)`
"""
function ChainRulesCore.rrule(::typeof(loglikelihood), (; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T}, N}, model::AbstractArray{T, N}) where {T, N}
    size(weighteddata) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_value(weighteddata)
    p = get_precision(weighteddata)

    AK.get_backend(data) == AK.get_backend(model) || error("scaledlikelihood : data and model must be on the same backend")

    a = similar(data)
    b = similar(data)
    AK.map(eachindex(data), AK.get_backend(data); init = zero(T)) do i
#    @inbounds @simd for i in eachindex(data, model)
        b[i] = model[i] .* p[i] .* data[i]
        a[i] = model[i] .* p[i] .* model[i]
    end

    α = sum(b, dims = dims) ./ sum(a, dims = dims)

    AK.map!(α,α) do αi
        ifelse((nonnegative && αi < 0) || !isfinite(αi), zero(T), αi)
    end
    r = (α .* model .- data)
    rp = r .* p
    loglikelihood_pullback(Δy) = (NoTangent(), NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, loglikelihood_pullback
end

end