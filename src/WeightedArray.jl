## Array of WeightedValue

"""
    value(x::AbstractArray{<:WeightedValue})

Extract an array with the values of each `WeightedValue` element.
"""
value(x::AbstractArray{<:WeightedValue}) = map(x -> x.value, x)

"""
    precision(x::AbstractArray{<:WeightedValue})

Extract an array with the precisions of each `WeightedValue` element.
"""
precision(x::AbstractArray{<:WeightedValue}) = map(x -> x.precision, x)

"""
    var(x::AbstractArray{<:WeightedValue})

Return the element-wise variance array of `x`, defined as the inverse
precision at each position:

`var(x) = 1 ./ precision(x)`.
"""
var(x::AbstractArray{<:WeightedValue}) = inv.(precision(x))

"""
    std(x::AbstractArray{<:WeightedValue})

Return the element-wise standard deviation array of `x`, defined as:

`std(x) = sqrt.(var(x))`.
"""
std(x::AbstractArray{<:WeightedValue}) = sqrt.(var(x))

WeightedValue(A::AbstractArray{T1, N}, B::AbstractArray{T2, N}) where {T1, T2, N} = map((a, b) -> WeightedValue(a, b), A, B)


"""
    WeightedArray{T,N}

`N`-dimensional array of `WeightedValue{T}`.
"""
const WeightedArray{T, N} = ZippedArray{WeightedValue{T}, N, 2, I, Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractArray{T, N}, I}

_WeightedArray(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T <: Real, N} = ZippedArray{WeightedValue{T}}(A, B)

"""
    WeightedArray(values::AbstractArray, precision::AbstractArray)

Build a weighted array from value and precision arrays of the same shape.
"""
function WeightedArray(A::AbstractArray{<:Union{Missing, T1}, N}, B::AbstractArray{T2, N}) where {T1 <: Real, T2 <: Real, N}
    size(A) == size(B) || error("WeightedArray: value and precision arrays must have the same shape")
    A, B = filterbaddata(A, B)
    return _WeightedArray(A, B)
end

WeightedArray(x::AbstractArray{<:WeightedValue}) = WeightedArray(value(x), precision(x))
WeightedArray(x::WeightedArray) = x

WeightedArray(x::AbstractArray{Missing}) = _WeightedArray(zeros(size(x)), zeros(size(x)))


function filterbaddata(val::AbstractArray{<:Union{Missing, T1}, N}, pre::AbstractArray{<:Union{Missing, T2}, N}) where {T1, T2, N}
    ((T = promote_type(T1, T2)) <: Real) || error("filterbaddata: value and precision arrays must have a real element type")
    map!(p -> ifelse(ismissing(p), T(NaN), p), pre, pre)
    map!(v -> ifelse(ismissing(v), T(NaN), v), val, val)

    prec = map((v, p) -> ifelse(isfinite(p) && isfinite(v), T(p), T(0)), val, pre)
    val = map((v, p) -> ifelse(isfinite(p) && isfinite(v), T(v), T(0)), val, pre)
    return val, prec
end


TypeUtils.get_precision(::Type{<:WeightedArray{T}}) where {T} = T


Base.zeros(::Type{WeightedValue{T1}}, dims::Int...) where {T1 <: Real} = WeightedArray(zeros(T1, dims...), fill(T1(+Inf), dims...))

Base.:+(A::WeightedArray, B::WeightedArray) = WeightedArray(A.value .+ B.value, inv.(inv.(A.precision) .+ inv.(B.precision)))
Broadcast.broadcasted(::typeof(+), A::WeightedArray, B::WeightedArray) = A + B
Base.:-(A::WeightedArray, B::WeightedArray) = WeightedArray(A.value .- B.value, inv.(inv.(A.precision) .+ inv.(B.precision)))
Broadcast.broadcasted(::typeof(-), A::WeightedArray, B::WeightedArray) = A - B

Base.:+((; value, precision)::WeightedArray, B::Union{T, AbstractArray{T}}) where {T <: Real} = WeightedArray(value .+ B, precision)
Broadcast.broadcasted(::typeof(+), A::WeightedArray, B::Union{T, AbstractArray{T}}) where {T <: Real} = A + B
Base.:+(A::Union{T, AbstractArray{T}}, B::WeightedArray) where {T <: Real} = B + A
Broadcast.broadcasted(::typeof(+), A::Union{T, AbstractArray{T}}, B::WeightedArray) where {T <: Real} = B + A
Base.:-((; value, precision)::WeightedArray, B::Union{T, AbstractArray{T}}) where {T <: Real} = WeightedArray(value .- B, precision)
Broadcast.broadcasted(::typeof(-), A::WeightedArray, B::Union{T, AbstractArray{T}}) where {T <: Real} = A - B
Base.:-(A::Union{T, AbstractArray{T}}, (; value, precision)::WeightedArray) where {T <: Real} = WeightedArray(A .- value, precision)
Broadcast.broadcasted(::typeof(-), A::Union{T, AbstractArray{T}}, B::WeightedArray) where {T <: Real} = A - B

Base.:/((; value, precision)::WeightedArray, B::Number) = WeightedArray(value ./ B, B^2 .* precision)
Broadcast.broadcasted(::typeof(/), A::WeightedArray, B::Number) = A / B
Broadcast.broadcasted(::typeof(/), (; value, precision)::WeightedArray, B::AbstractArray{<:Real}) = WeightedArray(value ./ B, B .^ 2 .* precision)
Base.:/(A::Number, (; value, precision)::WeightedArray) = WeightedArray(A ./ value, inv.(precision) ./ A .^ 2)
Broadcast.broadcasted(::typeof(/), A::Number, B::WeightedArray) = A / B
Broadcast.broadcasted(::typeof(/), A::AbstractArray{<:Real}, (; value, precision)::WeightedArray) = WeightedArray(A ./ value, inv.(precision) ./ A .^ 2)

Base.:*(B::Number, (; value, precision)::WeightedArray) = WeightedArray(value .* B, precision ./ B .^ 2)
Broadcast.broadcasted(::typeof(*), B::Number, A::WeightedArray) = B * A
Broadcast.broadcasted(::typeof(*), B::AbstractArray{<:Real}, (; value, precision)::WeightedArray) = WeightedArray(value .* B, precision ./ B .^ 2)
Base.:*(A::WeightedArray, B::Number) = B * A
Broadcast.broadcasted(::typeof(*), A::WeightedArray, B::Number) = B * A
Broadcast.broadcasted(::typeof(*), A::WeightedArray, B::AbstractArray{<:Real}) = B .* A

Base.:*(::WeightedArray, ::WeightedArray) = error("Multiplication of WeightedArray objects is not supported")
Base.:/(::WeightedArray, ::WeightedArray) = error("Division of WeightedArray objects is not supported")

Base.view(A::WeightedArray, I...) = WeightedArray(view(value(A), I...), view(precision(A), I...))

value(x::WeightedArray) = x.args[1]
precision(x::WeightedArray) = x.args[2]

Base.reshape(A::WeightedArray, dims::Union{Colon, Int64}...) = WeightedArray(reshape(value(A), dims...), reshape(precision(A), dims...))
Base.reshape(A::WeightedArray, ::Colon) = WeightedArray(reshape(value(A), :), reshape(precision(A), :))

Base.propertynames(::WeightedArray) = (:value, :precision)
function Base.getproperty(A::WeightedArray, s::Symbol)
    if s == :value
        return value(A)
    elseif s == :precision
        return precision(A)
    else
        getfield(A, s)
    end
end


function Base.summary(io::IO, A::WeightedArray{T, N}) where {T, N}
    shape = N == 1 ? "$(length(A))-element" : Base.dims2string(size(A))
    return print(io, shape, " WeightedArray{", T, ", ", N, "} (alias of ", typeof(A), "):")
end
