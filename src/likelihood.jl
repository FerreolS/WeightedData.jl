
struct L2Loss end

"""
    (loss::L2Loss)((; data, precision)::WeightedValue, model::Number)

Calculate the loss (negloglikelihood) for a weighted data point under a Gaussian model.

## Arguments
- `data`: The observed value in the weighted point.
- `precision`: The precision (inverse variance) associated with the observation.
- `model`: The predicted value from the model.

## Returns
The loss contribution: `(data - model)^2 * precision / 2`
"""
function (::L2Loss)((; value, precision)::WeightedValue, model::Number)
    return (value - model)^2 * precision / 2
end



"""
    likelihood(data::WeightedValue, model; loss=L2Loss())

Calculate the likelihood for a weighted data point.

## Arguments
- `data`: The observed value in the weighted point.
- `model`: The predicted value from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The neg log likelihood value.
"""
function likelihood(data::WeightedValue, model; loss=L2Loss())
    return likelihood(loss, data, model)
end


likelihood(loss, data::WeightedValue, model::Number) = loss(data, model)


"""
    get_weight(data::WeightedValue, model; loss=L2Loss())

Calculate the equivalent weight for a given the loss function for IRLS.

## Arguments
- `data`: The observed value in the weighted point.
- `model`: The predicted value from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The equivalent weight.
"""
function get_weight(data::WeightedValue, model; loss=L2Loss())
    return get_weight(loss, data, model)
end

get_weight(_, data::WeightedValue, _::Number) = get_precision(data)


"""
    likelihood(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss=L2Loss())

Calculate the likelihood for an array of weighted data points.

## Arguments
- `data`: The observed values in the weighted points.
- `model`: The predicted values from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The neg log likelihood value.
"""
function likelihood(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss=L2Loss())
    return likelihood(loss, data, model)
end

function likelihood(loss, data::AbstractArray{WeightedValue{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return mapreduce(loss, +, data, model)
end



function get_weight(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss=L2Loss())
    return get_weight(loss, data, model)
end

function get_weight(_, data::AbstractArray{WeightedValue{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return get_precision(data)
end




function ChainRulesCore.rrule(::typeof(likelihood), ::L2Loss, data::AbstractArray{WeightedValue{T},N}, model::AbstractArray{T,N}) where {T,N}
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

struct ScaledL2Loss
    dims::Int
    nonnegative::Bool
end
ScaledL2Loss(; dims=1, nonnegative=false) = ScaledL2Loss(dims, nonnegative)

function likelihood((; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(weighteddata) == size(model) || error("scaledL2loss : size(A) != size(model)")
    data = get_value(weighteddata)
    precision = get_precision(weighteddata)

    α = sum(model .* precision .* data, dims=dims) ./ sum(model .* precision .* model, dims=dims)

    if nonnegative
        α = max.(0, α)
    end

    α[.!isfinite.(α)] .= T2(0)
    res = (α .* model .- data)
    return sum(res .^ 2 .* precision) / 2
end

function ChainRulesCore.rrule(::typeof(likelihood), (; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(weighteddata) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_value(weighteddata)
    precision = get_precision(weighteddata)

    α = sum(model .* precision .* data, dims=dims) ./ sum(model .* precision .* model, dims=dims)

    if nonnegative
        α = max.(0, α)
    end
    α[.!isfinite.(α)] .= T2(0)

    r = (α .* model .- data)
    rp = r .* precision
    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end
