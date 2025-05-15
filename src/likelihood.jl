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
struct L2Loss end

function (::L2Loss)((; data, precision)::WeightedPoint, model::Number)
    return (data - model)^2 * precision / 2
end

l2loss() = L2Loss()

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

struct CauchyLoss{T}
    s::T
end

function (lkl::CauchyLoss)(data::WeightedPoint{T1}, model::T2) where {T1,T2}
    T = promote_type(T1, T2)
    lkl(convert(T, data), convert(T, model))
    return
end

function ((; s)::CauchyLoss)((; data, precision)::WeightedPoint{T}, model::T) where {T}
    C = T(s / 2.385)^2
    r = (model - data)
    return 1 / (2C) * log(T(1) + C * precision * r^2)
end

cauchyloss(s::Number) = CauchyLoss(s)


function likelihood(data::WeightedPoint, model; loss=L2Loss())
    return likelihood(loss, data, model)
end


likelihood(loss, data::WeightedPoint, model::Number) = loss(data, model)


function likelihood(data::AbstractArray{<:WeightedPoint}, model::AbstractArray; loss=l2loss())
    return likelihood(loss, data, model)
end

function likelihood(loss, data::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return mapreduce(loss, +, data, model)
end




function ChainRulesCore.rrule(::typeof(likelihood), ::L2Loss, data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray{T,N}) where {T,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")

    d = get_data(data)
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



function ChainRulesCore.rrule(::typeof(likelihood), (; s)::CauchyLoss, data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray{T2,N}) where {T,T2,N}
    C = T(s / 2.385)^2
    r = model .- get_data(data)
    rp = get_precision(data) .* r

    q = T(1) .+ C .* rp .* r

    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp ./ q .* Δy)
    return (1 / (2C)) .* sum(log, q), likelihood_pullback
end

function scaledL2loss(weighteddata::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(weighteddata) == size(model) || error("scaledL2loss : size(A) != size(model)")
    data = get_data(weighteddata)
    precision = get_precision(weighteddata)

    α = max.(0, sum(model .* precision .* data, dims=2) ./ sum(model .* precision .* model, dims=2))
    α[.!isfinite.(α)] .= T2(0)
    res = (α .* model .- data)
    return sum(res .^ 2 .* precision) / 2
end

function ChainRulesCore.rrule(::typeof(scaledL2loss), weighteddata::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(weighteddata) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    data = get_data(weighteddata)
    precision = get_precision(weighteddata)

    α = max.(0, sum(model .* precision .* data, dims=2) ./ sum(model .* precision .* model, dims=2))

    r = (α .* model .- data)
    rp = r .* precision
    likelihood_pullback(Δy) = (NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end
