module WeightedDataChainRulesCoreExt

import ChainRulesCore: NoTangent, ZeroTangent
import ChainRulesCore
import StatsAPI: loglikelihood
import WeightedData: L2Loss, get_value, get_precision, WeightedValue, ScaledL2Loss, oncpu
import Adapt: adapt
import TypeUtils: parameterless
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
    
    idx = oncpu(model) ?  eachindex(model) : adapt(parameterless(typeof(model)),collect(eachindex(model))) 
    l = mapreduce(+, idx; init = zero(T)) do i
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
function ChainRulesCore.rrule(::typeof(loglikelihood), (; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T}, N}, model::AbstractArray{T, N}) where {T, N}
    size(weighteddata) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    d = get_value(weighteddata)
    p = get_precision(weighteddata)


    a = similar(d)
    b = similar(d)

    idx = oncpu(model) ?  eachindex(model) : adapt(parameterless(typeof(model)),collect(eachindex(model))) 
    map(idx) do i
        b[i] = model[i] .* p[i] .* d[i]
        a[i] = model[i] .* p[i] .* model[i]
    end

    α = sum(b, dims = dims) ./ sum(a, dims = dims)

    map!( αi -> (((nonnegative && αi < 0) || !isfinite(αi)) ? zero(T) : αi) ,α, α)
    am  = α .* model
    l = mapreduce(+, idx; init = zero(T)) do i
        r = am[i] - d[i]
        a[i] = p[i] * r
        return r .* a[i]
    end 

    loglikelihood_pullback(Δy) = (NoTangent(), NoTangent(), ZeroTangent(), α .* a .* Δy)
    return l / 2, loglikelihood_pullback
end

end
