## Array of WeightedValue

"""
    get_value(x::AbstractArray{<:WeightedValue})

Extract an array with the values of each `WeightedValue` element.
"""
get_value(x::AbstractArray{<:WeightedValue}) = map(x -> x.value, x)

"""
    get_precision(x::AbstractArray{<:WeightedValue})

Extract an array with the precisions of each `WeightedValue` element.
"""
get_precision(x::AbstractArray{<:WeightedValue}) = map(x -> x.precision, x)
WeightedValue(A::AbstractArray{T1, N}, B::AbstractArray{T2, N}) where {T1, T2, N} = map((a, b) -> WeightedValue(a, b), A, B)


"""
    WeightedArray{T,N}

`N`-dimensional array of `WeightedValue{T}`.
"""
const WeightedArray{T, N} = ZippedArray{WeightedValue{T}, N, 2, I, Tuple{A, B}} where {A <: AbstractArray{T, N}, B <: AbstractArray{T, N}, I}

"""
    WeightedArray(values::AbstractArray{T,N}, precision::AbstractArray{T,N})

Build a weighted array from value and precision arrays of the same shape.
"""
WeightedArray(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T <: Real, N} = ZippedArray{WeightedValue{T}}(A, B)
WeightedArray(A::AbstractArray{T1, N}, B::AbstractArray{T2, N}) where {T1 <: Real, T2 <: Real, N} = ZippedArray{WeightedValue{T1}}(A, T1.(B))

"""
    WeightedArray(values::AbstractArray{<:Union{Missing,Real},N}, precision::AbstractArray{<:Real,N})

Build a `WeightedArray` while handling invalid entries.

Any position where `values` is `missing`/`NaN` or `precision` is `NaN` is
replaced by `(0, 0)` in the resulting weighted array.
"""
function WeightedArray(x::AbstractArray{<:Union{Missing, Real}, N}, y::AbstractArray{T2, N}) where {T2 <: Real, N}
    size(x) == size(y) || error("WeightedArray: size(value) != size(precision)")
    Tx = Base.nonmissingtype(eltype(x))
    T = Tx <: Real ? promote_type(Tx, T2) : T2
    value = map(x, y) do xv, yv
        (ismissing(xv) || isnan(xv) || isnan(yv)) ? zero(T) : T(xv)
    end
    precision = map(x, y) do xv, yv
        (ismissing(xv) || isnan(xv) || isnan(yv)) ? zero(T) : T(yv)
    end
    return ZippedArray{WeightedValue{T}}(value, precision)
end


WeightedArray(x::AbstractArray{<:WeightedValue}) = WeightedArray(get_value(x), get_precision(x))
WeightedArray(x::WeightedArray) = x

WeightedArray(x::AbstractArray{Missing}) = WeightedArray(zeros(size(x)), zeros(size(x)))
#WeightedArray(x::AbstractArray{T}) where {T <: Real} = WeightedArray(x, ones(size(x)))
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

Base.view(A::WeightedArray, I...) = WeightedArray(view(get_value(A), I...), view(get_precision(A), I...))

get_value(x::WeightedArray) = x.args[1]
get_precision(x::WeightedArray) = x.args[2]

Base.reshape(A::WeightedArray, dims::Union{Colon, Int64}...) = WeightedArray(reshape(get_value(A), dims...), reshape(get_precision(A), dims...))
Base.reshape(A::WeightedArray, ::Colon) = WeightedArray(reshape(get_value(A), :), reshape(get_precision(A), :))

Base.propertynames(::WeightedArray) = (:value, :precision)
function Base.getproperty(A::WeightedArray, s::Symbol)
    if s == :value
        return get_value(A)
    elseif s == :precision
        return get_precision(A)
    else
        getfield(A, s)
    end
end

function flagbadpix(data::AbstractArray{WeightedValue{T}, N}, badpix::Union{Array{Bool, N}, BitArray{N}}) where {T, N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return @inbounds map((d, flag) -> ifelse(flag, WeightedValue(T(0), T(0)), d), data, badpix)
end

"""
    flagbadpix!(data::WeightedArray, badpix)

In-place version of `flagbadpix` for `WeightedArray`.

All positions where `badpix` is `true` are set to `(0, 0)`.
"""
function flagbadpix!(data::WeightedArray{T1, N}, badpix::Union{AbstractArray{Bool, N}, BitArray{N}}) where {T1, N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return data[badpix] .= WeightedValue{T1}(T1(0), T1(0))
end
