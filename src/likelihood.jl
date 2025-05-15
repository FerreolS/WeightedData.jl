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



function likelihood(data::WeightedPoint, model; loss=L2Loss())
    return likelihood(loss, data, model)
end


likelihood(loss, data::WeightedPoint, model::Number) = loss(data, model)


function get_weight(data::WeightedPoint, model; loss=L2Loss())
    return get_weight(loss, data, model)
end

get_weight(_, data::WeightedPoint, _::Number) = get_precision(data)


function likelihood(data::AbstractArray{<:WeightedPoint}, model::AbstractArray; loss=l2loss())
    return likelihood(loss, data, model)
end

function likelihood(loss, data::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return mapreduce(loss, +, data, model)
end



function get_weight(data::AbstractArray{<:WeightedPoint}, model::AbstractArray; loss=l2loss())
    return get_weight(loss, data, model)
end

function get_weight(_, data::AbstractArray{WeightedPoint{T1},N}, model::AbstractArray{T2,N}) where {T1,T2,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return get_precision(data)
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
