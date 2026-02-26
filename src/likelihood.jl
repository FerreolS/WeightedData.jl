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
    loglikelihood(data::WeightedValue, model; loss=L2Loss())

Calculate the negative log-likelihood for a weighted data point.

## Arguments
- `data`: The observed value in the weighted point.
- `model`: The predicted value from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The neg log likelihood value.

## Deprecation note
`likelihood(data, model; loss=...)` is deprecated and forwards to this method.
"""
function loglikelihood(data::WeightedValue, model; loss = L2Loss())
    return loglikelihood(loss, data, model)
end


loglikelihood(loss, data::WeightedValue, model::Number) = loss(data, model)


"""
    get_weights(data::WeightedValue, model; loss=L2Loss())

Calculate the equivalent weight for a given the loss function for IRLS.

## Arguments
- `data`: The observed value in the weighted point.
- `model`: The predicted value from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The equivalent weight.
"""
function get_weights(data::WeightedValue, model; loss = L2Loss())
    return get_weights(loss, data, model)
end

get_weights(_, data::WeightedValue, _::Number) = precision(data)


"""
    loglikelihood(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss=L2Loss())

Calculate the negative log-likelihood for an array of weighted data points.

## Arguments
- `data`: The observed values in the weighted points.
- `model`: The predicted values from the model.
- `loss`: The loss function to use (default: L2 loss).

## Returns
The neg log likelihood value.

## Deprecation note
`likelihood(data, model; loss=...)` is deprecated and forwards to this method.
"""
function loglikelihood(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss = L2Loss())
    return loglikelihood(loss, data, model)
end

function loglikelihood(loss, data::AbstractArray{WeightedValue{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(data) == size(model) || error("loglikelihood : size(A) != size(model)")
    return mapreduce(loss, +, data, model)
end

function get_weights(data::AbstractArray{<:WeightedValue}, model::AbstractArray; loss = L2Loss())
    return get_weights(loss, data, model)
end

function get_weights(_, data::AbstractArray{WeightedValue{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(data) == size(model) || error("get_weights : size(A) != size(model)")
    return precision(data)
end

struct ScaledL2Loss
    dims::Int
    nonnegative::Bool
end
ScaledL2Loss(; dims = 1, nonnegative = false) = ScaledL2Loss(dims, nonnegative)

function loglikelihood((; dims, nonnegative)::ScaledL2Loss, weighteddata::AbstractArray{WeightedValue{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(weighteddata) == size(model) || error("scaledL2loss : size(A) != size(model)")
    data = value(weighteddata)
    p = precision(weighteddata)


    a = similar(data)
    b = similar(data)
    @inbounds @simd for i in eachindex(data, model)
        b[i] = model[i] .* p[i] .* data[i]
        a[i] = model[i] .* p[i] .* model[i]
    end

    α = sum(b, dims = dims) ./ sum(a, dims = dims)

    @inbounds @simd for i in eachindex(α)
        if (nonnegative && α[i] < 0) || !isfinite(α[i])
            α[i] = T2(0)
        end
    end
    res = @. (α * model - data)^2 * p
    return sum(res) / 2
end

@deprecate likelihood(args...; kwargs...) loglikelihood(args...; kwargs...)
