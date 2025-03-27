"""
    WeightedPoint{T<:AbstractFloat} <: Number

A structure representing a numerical value weighted by its precision.

## Fields
- `val::T`: The value
- `precision::T`: The precision (or weight) associated with the value (must be positive)

## Examples
```julia
x = WeightedPoint(1.0, 0.5)  # value 1.0 with precision 0.5
```
"""
struct WeightedPoint{T<:AbstractFloat} <: Number
    val::T
    precision::T
    function WeightedPoint(val::T, precision::T) where {T<:AbstractFloat}
        precision >= 0 || error("WeightedPoint : precisionmust be positive")
        isfinite(val) || return new{T}(zero(T), zero(T))
        return new{T}(val, precision)
    end
    WeightedPoint{T}(val::T, precision::T) where {T<:AbstractFloat} = new{T}(val, precision)
end

WeightedPoint(val::T, precision::Number) where {T<:AbstractFloat} = WeightedPoint(val, T(precision))
WeightedPoint{T}(val::Number, precision::Number) where {T} = WeightedPoint(T(val), T(precision))

Base.real(A::WeightedPoint) = WeightedPoint(real(A.val), real(A.precision))
Base.imag(A::WeightedPoint) = WeightedPoint(imag(A.val), imag(A.precision))

"""
    +(A::WeightedPoint, B::WeightedPoint)

Add two weighted points. The result's value is the sum of values, and the precision
is calculated according to error propagation rules.
"""
Base.:+(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.val + B.val, inv(inv(A.precision) + inv(B.precision)))
Base.:+((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val + B, precision)
Base.:+(A::Number, B::WeightedPoint) = B + A
Base.:-(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.val - B.val, inv(inv(A.precision) + inv(B.precision)))
Base.:-((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val - B, precision)
Base.:/((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val / B, B^2 * precision)
Base.:/(A::Number, (; val, precision)::WeightedPoint) = WeightedPoint(A / val, inv(precision) / A^2)
Base.:*(B::Number, (; val, precision)::WeightedPoint) = WeightedPoint(val * B, precision / B^2)
Base.:*(A::WeightedPoint, B::Number) = B * A
Base.:*(::WeightedPoint, ::WeightedPoint) = error("Multiplication of WeightedPoint objects is not supported")
Base.:/(::WeightedPoint, ::WeightedPoint) = error("Division of WeightedPoint objects is not supported")

Base.one(::WeightedPoint{T}) where {T} = one(T)
Base.zero(::WeightedPoint{T}) where {T} = WeightedPoint(zero(T), T(+Inf))
Base.zero(::Type{WeightedPoint{T}}) where {T} = WeightedPoint(zero(T), T(+Inf))

Base.:(==)(x::WeightedPoint, y::WeightedPoint) = x.val == y.val && x.precision == y.precision

#Base.show(io::IO, x::WeightedPoint) = print(io, "WeightedPoint($(x.val), $(x.precision))")

Base.convert(::Type{T}, (; val, precision)::WeightedPoint) where {T<:Real} = WeightedPoint(convert(T, val), convert(T, precision))
""" 
    combine(A::WeightedPoint, B::WeightedPoint)

Combines two WeightedPoint objects by calculating their weighted average based on precision. The result has a precision equal to the sum of the individual precisions.

Example
```julia
    x = WeightedPoint(1.0, 0.5)
    y = WeightedPoint(2.0, 1.5)
    z = combine(x, y)  # WeightedPoint(1.75, 2.0)
```
"""
function combine(A::WeightedPoint, B::WeightedPoint)
    precision = A.precision + B.precision
    val = (A.precision * A.val + B.precision * B.val) / (precision)
    return WeightedPoint(val, precision)
end
""" 
    combine(B::NTuple{N,WeightedPoint{T}}) where {N,T}

Combines a tuple of WeightedPoint objects by calculating their weighted average. """
combine(B::NTuple{N,WeightedPoint{T}}) where {N,T} = combine(first(B), last(B, N - 1)...)
combine() = zero(WeightedPoint{Float64})
combine(A::WeightedPoint, B...) = combine(combine(A, first(B)), last(B, length(B) - 1)...)
combine(A::WeightedPoint) = A
