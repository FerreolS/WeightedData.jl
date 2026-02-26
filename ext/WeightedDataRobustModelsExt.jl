module WeightedDataRobustModelsExt

import WeightedData: likelihood, WeightedValue, get_value, get_precision, get_weight

import RobustModels: LossFunction,
    BoundedLossFunction,
    ConvexLossFunction,
    CompositeLossFunction,
    L2Loss,
    L1Loss,
    HuberLoss,
    L1L2Loss,
    FairLoss,
    LogcoshLoss,
    ArctanLoss,
    CatoniWideLoss,
    CatoniNarrowLoss,
    CauchyLoss,
    GemanLoss,
    WelschLoss,
    TukeyLoss,
    YohaiZamarLoss,
    HardThresholdLoss,
    HampelLoss,
    loss,
    rho,
    weight,
    workingweights


"""
    likelihood(loss::LossFunction, data::WeightedValue, model::Number)

Compute robust loss contribution for a single weighted observation.

Residual is defined as:
`r = sqrt(get_precision(data)) * (model - get_value(data))`
and the returned value is `rho(loss, r)`.
"""
function likelihood(loss::LossFunction, data::WeightedValue, model::Number)
    r = sqrt(get_precision(data)) * (model - get_value(data))
    return rho(loss, r)
end

"""
    likelihood(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray)

Compute robust loss for arrays of weighted observations.
`data` and `model` must have the same shape.
"""
function likelihood(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray)
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    r = @. sqrt($get_precision(data)) * (model - $get_value(data))
    l = Base.Fix1(rho, loss)
    return mapreduce(l, +, r)
end

"""
    get_weight(loss::LossFunction, data::WeightedValue, model::Number)

Compute IRLS weight for a single weighted observation.
"""
function get_weight(loss::LossFunction, data::WeightedValue, model::Number)
    r = sqrt(get_precision(data)) * (model - get_value(data))
    return weight(loss, r)
end

"""
    get_weight(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray)

Compute IRLS weights element-wise for arrays of weighted observations.
`data` and `model` must have the same shape.
"""
function get_weight(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray)
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    r = @. sqrt($get_precision(data)) * (model - $get_value(data))
    w = Base.Fix1(weight, loss)
    return map(w, r)
end

"""
    workingweights(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray)

Compute the working weights for a given loss function, weighted data, and model.

This function calculates the weights used in iterative weighted least squares algorithms
by delegating to [`get_weight`](@ref) with the specified loss function, data, and model parameters.

# Arguments
- `loss::LossFunction`: The loss function defining the weighting scheme
- `data::AbstractArray{<:WeightedValue}`: Array of weighted data points
- `model::AbstractArray`: Model parameters or predictions

# Returns
Array of working weights corresponding to the input data

# See Also
- [`get_weight`](@ref)
- `RobustModels.LossFunction`
- [`WeightedValue`](@ref)
"""
workingweights(loss::LossFunction, data::AbstractArray{<:WeightedValue}, model::AbstractArray) = get_weight(loss, data, model)

end

#= 

function ChainRulesCore.rrule(::typeof(likelihood), (; s)::CauchyLoss, data::AbstractArray{WeightedValue{T},N}, model::AbstractArray{T2,N}) where {T,T2,N}
    C = T(s / 2.385)^2
    r = model .- get_value(data)
    rp = get_precision(data) .* r

    q = T(1) .+ C .* rp .* r

    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp ./ q .* Δy)
    return (1 / (2C)) .* sum(log, q), likelihood_pullback
end =#
