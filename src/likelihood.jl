"""
    gausslikelihood((; data, precision)::WeightedPoint{T}, model::Number) where {T}

Calculate the negative log-likelihood for a weighted data point under a Gaussian model.

# Arguments
- `data`: The observed value in the weighted point.
- `precision`: The precision (inverse variance) associated with the observation.
- `model`: The predicted value from the model.

# Returns
The negative log-likelihood contribution: `(data - model)^2 * precision / 2`
"""
function gausslikelihood((; data, precision)::WeightedPoint{T}, model::Number) where {T}
    return (data - model)^2 * precision / 2
end

"""
    robustlikelihood((; data, precision)::WeightedPoint{T1}, model::T2, s::Number) where {T1,T2}

Compute the robust likelihood for a given weighted data point, model, and scale parameter.

# Arguments
- `(; data, precision)::WeightedPoint{T1}`: A weighted data point
- `model::T2`: The model value to compare against the data point.
- `s::Number`: The scale parameter.

# Returns
- the robust likelihood value.

# Details
The function calculates the robust likelihood using the formula:
- `γ = 2.385`
- `r = (s / γ) * sqrt(precision) * (model - data)`
- `log(1 + r^2)`

The scale parameter `s` is used to control the robustness of the likelihood calculation.
"""
function robustlikelihood((; data, precision)::WeightedPoint{T1}, model::T2, s::Number) where {T1,T2}
    T = promote_type(T1, T2)
    γ = T(2.385)
    r = T(s / γ) * sqrt(precision) * (model - data)
    return log(T(1) + r^2)
end

robustlikelihood(s::Number) = (D::WeightedPoint, model::Number) -> robustlikelihood(D, model, s)


function likelihood(data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray; likelihoodfunc::F=gausslikelihood) where {T,N,F<:Function}
    return likelihood(likelihoodfunc, data, model)
end
function likelihood(likelihoodfunc::F, data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray{T2,N}) where {T,T2,N,F<:Function}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return mapreduce(likelihoodfunc, +, data, model)
end




function ChainRulesCore.rrule(::typeof(WeightedData.likelihood), ::typeof(WeightedData.gausslikelihood), data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray{T2,N}) where {T,T2,N}
    r = model .- get_data(data)
    rp = get_precision(data) .* r
    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end


function scaledlikelihood(data::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_data(data)
    precision = get_precision(data)

    α = max.(0, sum(model .* precision .* data, dims=2) ./ sum(model .* precision .* model, dims=2))
    α[.!isfinite.(α)] .= T2(0)
    res = (α .* model .- data)
    return sum(res .^ 2 .* precision) / 2
end

function ChainRulesCore.rrule(::typeof(scaledlikelihood), data::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_data(data)
    precision = get_precision(data)

    α = max.(0, sum(model .* precision .* data, dims=2) ./ sum(model .* precision .* model, dims=2))

    r = (α .* model .- data)
    rp = r .* precision
    likelihood_pullback(Δy) = (NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end
