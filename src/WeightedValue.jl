"""
    WeightedValue{T<:Real} <: Number

A structure representing a numerical value weighted by its precision.

## Fields
- `value::T`: The value
- `precision::T`: The precision (or weight) associated with the value (must be positive)

## Examples
```julia
x = WeightedValue(1.0, 0.5)  # value 1.0 with precision 0.5
```
"""
struct WeightedValue{T <: Real} <: Number
    value::T
    precision::T
    function WeightedValue(value::T, precision::T) where {T <: Real}
        precision >= 0 || error("WeightedValue : precisionmust be positive")
        isfinite(value) || return new{T}(zero(T), zero(T))
        return new{T}(value, precision)
    end
    WeightedValue{T}(value::T, precision::T) where {T <: Real} = new{T}(value, precision)
end

WeightedValue(value::T, precision::Number) where {T <: Real} = WeightedValue(value, T(precision))
WeightedValue{T}(value::Number, precision::Number) where {T} = WeightedValue(T(value), T(precision))
WeightedValue(::Missing, x) = WeightedValue(0, 0)

get_value(A::WeightedValue) = A.value
get_precision(A::WeightedValue) = A.precision

Base.real(A::WeightedValue) = WeightedValue(real(A.value), real(A.precision))
Base.imag(A::WeightedValue) = WeightedValue(imag(A.value), imag(A.precision))

"""
    +(A::WeightedValue, B::WeightedValue)

Add two weighted points. The result's value is the sum of values, and the precision
is calculated according to error propagation rules.
"""
Base.:+(A::WeightedValue, B::WeightedValue) = WeightedValue(A.value + B.value, inv(inv(A.precision) + inv(B.precision)))
Base.:+((; value, precision)::WeightedValue, B::Number) = WeightedValue(value + B, precision)
Base.:+(A::Number, B::WeightedValue) = B + A
Base.:-(A::WeightedValue, B::WeightedValue) = WeightedValue(A.value - B.value, inv(inv(A.precision) + inv(B.precision)))
Base.:-((; value, precision)::WeightedValue, B::Number) = WeightedValue(value - B, precision)
Base.:/((; value, precision)::WeightedValue, B::Number) = WeightedValue(value / B, B^2 * precision)
Base.:/(A::Number, (; value, precision)::WeightedValue) = WeightedValue(A / value, inv(precision) / A^2)
Base.:*(B::Number, (; value, precision)::WeightedValue) = WeightedValue(value * B, precision / B^2)
Base.:*(A::WeightedValue, B::Number) = B * A
Base.:*(::WeightedValue, ::WeightedValue) = error("Multiplication of WeightedValue objects is not supported")
Base.:/(::WeightedValue, ::WeightedValue) = error("Division of WeightedValue objects is not supported")

Base.one(::WeightedValue{T}) where {T} = one(T)
Base.zero(::WeightedValue{T}) where {T} = zero(WeightedValue{T})
Base.zero(::Type{WeightedValue{T}}) where {T} = WeightedValue(zero(T), T(+Inf))

Base.:(==)(x::WeightedValue, y::WeightedValue) = x.value == y.value && x.precision == y.precision


Base.convert(::Type{T}, (; value, precision)::WeightedValue) where {T <: Real} = WeightedValue(convert(T, value), convert(T, precision))
""" 
    weightedmean(A::WeightedValue, B::WeightedValue)

weightedmeans two WeightedValue objects by calculating their weighted average based on precision. The result has a precision equal to the sum of the individual precisions.

Example
```julia
    x = WeightedValue(1.0, 0.5)
    y = WeightedValue(2.0, 1.5)
    z = weightedmean(x, y)  # WeightedValue(1.75, 2.0)
```
"""
function weightedmean(A::WeightedValue, B::WeightedValue)
    precision = A.precision + B.precision
    value = (A.precision * A.value + B.precision * B.value) / (precision)
    return WeightedValue(value, precision)
end
""" 
    weightedmean(B::NTuple{N,WeightedValue{T}}) where {N,T}

weightedmeans a tuple of WeightedValue objects by calculating their weighted average. """
weightedmean(A::NTuple{N, WeightedValue}) where {N} = reduce(weightedmean, A)
weightedmean(::NTuple{0}) = weightedmean()
weightedmean() = nothing #zero(WeightedValue{Float64})
#weightedmeandmean(A::WeightedValue, B...) = weightedmean(weightedmean(A, first(B)), last(B, length(B) - 1))
#weightedmean(A::WeightedValue, B::NTuple{N,WeightedValue}) where {N} = weightedmean(weightedmean(A, first(B)), last(B, N - 1))
weightedmean(A::WeightedValue) = A
weightedmean(A::WeightedValue...) = reduce(weightedmean, A)

Base.show(io::IO, (; value, precision)::WeightedValue) = print(io, "$value ± $(1 / sqrt(precision))")
