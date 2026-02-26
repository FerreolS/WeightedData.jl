"""
    WeightedValue{T<:Real} <: Number

Value with associated precision (inverse variance).

`WeightedValue` stores a numeric `value` and a non-negative `precision`.
It is the scalar building block used throughout `WeightedData.jl`.

Constructor behavior:
- `precision < 0` throws an error;
- non-finite input `value` is converted to `(0, 0)`.

## Fields
- `value::T`: The value
- `precision::T`: The precision (must be non-negative)

## Example
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
    function WeightedValue{T}(value::T, precision::T) where {T <: Real}
        precision >= 0 || error("WeightedValue : precisionmust be positive")
        isfinite(value) || return new{T}(zero(T), zero(T))
        return new{T}(value, precision)
    end
end

WeightedValue(value::T, precision::Number) where {T <: Real} = WeightedValue(value, T(precision))
WeightedValue{T}(value::Number, precision::Number) where {T} = WeightedValue(T(value), T(precision))
WeightedValue(::Missing, x) = WeightedValue(0, 0)
WeightedValue(::Missing) = WeightedValue(0, 0)

"""
    get_value(x::WeightedValue)

Return the numeric value stored in `x`.
"""
get_value(A::WeightedValue) = A.value

"""
    get_precision(x::WeightedValue)

Return the precision (inverse variance) stored in `x`.
"""
get_precision(A::WeightedValue) = A.precision

"""
    var(x::WeightedValue)

Return the variance of `x`, defined as the inverse precision:

`var(x) = 1 / get_precision(x)`.
"""
var(A::WeightedValue) = inv(get_precision(A))

"""
    std(x::WeightedValue)

Return the standard deviation of `x`, defined as:

`std(x) = sqrt(var(x))`.
"""
std(A::WeightedValue) = sqrt(var(A))

Base.real(A::WeightedValue) = WeightedValue(real(A.value), real(A.precision))
Base.imag(A::WeightedValue) = WeightedValue(imag(A.value), imag(A.precision))

"""
    +(A::WeightedValue, B::WeightedValue)

Add two weighted values.

The resulting value is `A.value + B.value`. The resulting precision is computed
using standard independent-error propagation:

`p = 1 / (1/A.precision + 1/B.precision)`
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

TypeUtils.get_precision(::Type{<:WeightedValue{T}}) where {T} = T

"""
    show(io::IO, x::WeightedValue)

Pretty-print `x` as `value ± uncertainty`, with uncertainty computed as
`1/sqrt(precision)`. The number of significant digits in the uncertainty can
be controlled with IO context key `:error_digits` (default: `2`).
"""
function Base.show(io::IO, (; value, precision)::WeightedValue)
    error_digits = get(io, :error_digits, 2)
    return print(io, value, " ± ", round(1 / sqrt(precision), sigdigits = error_digits))
end

"""
    show(io::IO, ::MIME"text/plain", x::WeightedValue)

Pretty-print `x` in non-compact mode with explicit type information.
"""
function Base.show(io::IO, ::MIME"text/plain", x::WeightedValue{T}) where {T}
    typeinfo = get(io, :typeinfo, Any)
    if !(get(io, :compact, false) || (typeinfo isa DataType && typeinfo <: WeightedValue))
        println(io, summary(x))
    end
    return show(io, x)
end
