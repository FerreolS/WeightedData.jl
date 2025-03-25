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
        precision >= 0 || error("WeightedPoint : precision < 0 ")
        return new{T}(val, precision)
    end
end

WeightedPoint(val::T, precision::Number) where {T<:AbstractFloat} = WeightedPoint(val, T(precision))

Base.real(A::WeightedPoint) = WeightedPoint(real(A.val), real(A.precision))
Base.imag(A::WeightedPoint) = WeightedPoint(imag(A.val), imag(A.precision))

Base.:+(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.val + B.val, 1 / (1 / A.precision + 1 / B.precision))
Base.:+((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val + B, precision)
Base.:+(A::Number, B::WeightedPoint) = B + A
Base.:-(A::WeightedPoint, B::WeightedPoint) = WeightedPoint(A.val - B.val, 1 / (1 / A.precision + 1 / B.precision))
Base.:-((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val - B, precision)
Base.:/((; val, precision)::WeightedPoint, B::Number) = WeightedPoint(val / B, B^2 * precision)
Base.:*(B::Number, (; val, precision)::WeightedPoint) = WeightedPoint(val * B, precision / B^2)
Base.:*(A::WeightedPoint, B::Number) = B * A

Base.:(==)(x::WeightedPoint, y::WeightedPoint) = x.val == y.val && x.precision == y.precision

Base.show(io::IO, x::WeightedPoint) = print(io, "WeightedPoint($(x.val), $(x.precision))")

Base.convert(::Type{T}, (; val, precision)::WeightedPoint) where {T<:Real} = convert(T, val)

function combine(A::WeightedPoint, B::WeightedPoint)
    precision = A.precision + B.precision
    val = (A.precision * A.val + B.precision * B.val) / (precision)
    return WeightedPoint(val, precision)
end
combine(B::NTuple{N,WeightedPoint{T}}) where {N,T} = combine(first(B), last(B, N - 1)...)
combine(A::WeightedPoint, B...) = combine(combine(A, first(B)), last(B, length(B) - 1)...)
combine(B::AbstractArray{WeightedPoint{T}}) where {T} = combine(first(B), last(B, length(B) - 1)...)
combine(A::WeightedPoint, B::AbstractArray{WeightedPoint{T}}) where {T} = combine(combine(A, first(B)), last(B, length(B) - 1)...)
combine(A::WeightedPoint) = A
