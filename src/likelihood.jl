function gausslikelihood((; val, precision)::WeightedPoint{T}, model::Number) where {T}
    return (val - model)^2 * precision / 2
end

function robustlikelihood((; val, precision)::WeightedPoint{T1}, model::T2, s::Number) where {T1, T2}
    T = promote_type(T1, T2)
    γ = T(2.385)
    r = T(s / γ) * sqrt(precision) * (model - val)
    return log(T(1) + r^2)
end

robustlikelihood(s::Number) = (D::WeightedPoint, model::Number) -> robustlikelihood(D, model, s)


function likelihood(data::AbstractArray{WeightedPoint{T}, N}, model::AbstractArray; likelihoodfunc::F = gausslikelihood) where {T, N, F <: Function}
    return likelihood(likelihoodfunc, data, model)
end
function likelihood(likelihoodfunc::F, data::AbstractArray{WeightedPoint{T}, N}, model::AbstractArray{T2, N}) where {T, T2, N, F <: Function}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    return mapreduce((x, y) -> likelihoodfunc(x, y), +, data, model)
end


function ChainRulesCore.rrule(::typeof(WeightedData.likelihood), ::typeof(WeightedData.gausslikelihood), data::AbstractArray{WeightedPoint{T}, N}, model::AbstractArray{T2, N}) where {T, T2, N}
    r = model .- get_val(data)
    rp = get_precision(data) .* r
    likelihood_pullback(Δy) = (NoTangent(), NoTangent(), NoTangent(), rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end


function scaledlikelihood(data::AbstractArray{WeightedPoint{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    val = get_val(data)
    precision = get_precision(data)

    α = max.(0, sum(model .* precision .* val, dims = 2) ./ sum(model .* precision .* model, dims = 2))
    α[.!isfinite.(α)] .= T2(0)
    res = (α .* model .- val)
    return sum(res .^ 2 .* precision) / 2
end

function ChainRulesCore.rrule(::typeof(scaledlikelihood), data::AbstractArray{WeightedPoint{T1}, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
    val = get_val(data)
    precision = get_precision(data)

    α = max.(0, sum(model .* precision .* val, dims = 2) ./ sum(model .* precision .* model, dims = 2))

    r = (α .* model .- val)
    rp = r .* precision
    likelihood_pullback(Δy) = (NoTangent(), ZeroTangent(), α .* rp .* Δy)
    return sum(r .* rp) / 2, likelihood_pullback
end
