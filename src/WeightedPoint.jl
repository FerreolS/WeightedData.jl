"""
    WeightedPoint{T<:AbstractFloat} <: Number

A structure representing a numerical value weighted by its precision.

## Fields
- `data::T`: The value
- `precision::T`: The precision (or weight) associated with the value (must be positive)

## Examples
```julia
x = WeightedPoint(1.0, 0.5)  # value 1.0 with precision 0.5
```
"""
struct WeightedPoint{T<:AbstractFloat} <: Number
    data::T
    precision::T
    function WeightedPoint(data::T, precision::T) where {T<:AbstractFloat}
        precision >= 0 || error("WeightedPoint : precisionmust be positive")
        isfinite(data) || return new{T}(zero(T), zero(T))
        return new{T}(data, precision)
    end
    WeightedPoint{T}(data::T, precision::T) where {T<:AbstractFloat} = new{T}(data, precision)
end

WeightedPoint(data::T, precision::Number) where {T<:AbstractFloat} = WeightedPoint(data, T(precision))
WeightedPoint{T}(data::Number, precision::Number) where {T} = WeightedPoint(T(data), T(precision))

get_data(A::WeightedPoint) = A.data
get_precision(A::WeightedPoint) = A.precision

Base.real(A::WeightedPoint) = WeightedPoint(real(A.data), real(A.precision))
Base.imag(A::WeightedPoint) = WeightedPoint(imag(A.data), imag(A.precision))

"""
    +(A::WeightedPoint, B::WeightedPoint)

Add two weighted points. The result's value is the sum of values, and the precision
is calculated according to error propagation rules.
"""
Base.:+(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.data + B.data, inv(inv(A.precision) + inv(B.precision)))
Base.:+((; data, precision)::WeightedPoint, B::Number) = WeightedPoint(data + B, precision)
Base.:+(A::Number, B::WeightedPoint) = B + A
Base.:-(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.data - B.data, inv(inv(A.precision) + inv(B.precision)))
Base.:-((; data, precision)::WeightedPoint, B::Number) = WeightedPoint(data - B, precision)
Base.:/((; data, precision)::WeightedPoint, B::Number) = WeightedPoint(data / B, B^2 * precision)
Base.:/(A::Number, (; data, precision)::WeightedPoint) = WeightedPoint(A / data, inv(precision) / A^2)
Base.:*(B::Number, (; data, precision)::WeightedPoint) = WeightedPoint(data * B, precision / B^2)
Base.:*(A::WeightedPoint, B::Number) = B * A
Base.:*(::WeightedPoint, ::WeightedPoint) = error("Multiplication of WeightedPoint objects is not supported")
Base.:/(::WeightedPoint, ::WeightedPoint) = error("Division of WeightedPoint objects is not supported")

Base.one(::WeightedPoint{T}) where {T} = one(T)
Base.zero(::WeightedPoint{T}) where {T} = zero(WeightedPoint{T})
Base.zero(::Type{WeightedPoint{T}}) where {T} = WeightedPoint(zero(T), T(+Inf))

Base.:(==)(x::WeightedPoint, y::WeightedPoint) = x.data == y.data && x.precision == y.precision

#Base.show(io::IO, x::WeightedPoint) = print(io, "WeightedPoint($(x.data), $(x.precision))")

Base.convert(::Type{T}, (; data, precision)::WeightedPoint) where {T<:Real} = WeightedPoint(convert(T, data), convert(T, precision))
""" 
    weightedmean(A::WeightedPoint, B::WeightedPoint)

weightedmeans two WeightedPoint objects by calculating their weighted average based on precision. The result has a precision equal to the sum of the individual precisions.

Example
```julia
    x = WeightedPoint(1.0, 0.5)
    y = WeightedPoint(2.0, 1.5)
    z = weightedmean(x, y)  # WeightedPoint(1.75, 2.0)
```
"""
function weightedmean(A::WeightedPoint, B::WeightedPoint)
    precision = A.precision + B.precision
    data = (A.precision * A.data + B.precision * B.data) / (precision)
    return WeightedPoint(data, precision)
end
""" 
    weightedmean(B::NTuple{N,WeightedPoint{T}}) where {N,T}

weightedmeans a tuple of WeightedPoint objects by calculating their weighted average. """
weightedmean(A::NTuple{N,WeightedPoint}) where {N} = reduce(weightedmean, A)
weightedmean(::NTuple{0}) = weightedmean()
weightedmean() = nothing #zero(WeightedPoint{Float64})
#weightedmeandmean(A::WeightedPoint, B...) = weightedmean(weightedmean(A, first(B)), last(B, length(B) - 1))
#weightedmean(A::WeightedPoint, B::NTuple{N,WeightedPoint}) where {N} = weightedmean(weightedmean(A, first(B)), last(B, N - 1))
weightedmean(A::WeightedPoint) = A
weightedmean(A::WeightedPoint...) = reduce(weightedmean, A)
