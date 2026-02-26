module WeightedDataUncertainExt
import Uncertain
import WeightedData: WeightedValue, value, precision

"""
    WeightedValue(x::Uncertain.Value)

Convert an `Uncertain.Value` to `WeightedValue`.

# Arguments
- `x::Uncertain.Value`: value with uncertainty

# Returns
A `WeightedValue` with:
- value = `Uncertain.value(x)`
- precision = `Uncertain.uncertainty(x)^(-2)`

# Notes
Precision is the inverse variance.

# Example
```julia
using Uncertain, WeightedData
u = Uncertain.Value(1.0, 0.1)  # value = 1.0, uncertainty = 0.1
w = WeightedValue(u)  # WeightedValue(1.0, 100.0)
```
"""
WeightedValue(x::Uncertain.Value) = WeightedValue(Uncertain.value(x), Uncertain.uncertainty(x)^(-2))

"""
    T(x::WeightedValue) where T <: Uncertain.Value

Convert `WeightedValue` to an `Uncertain.Value` subtype.

# Arguments
- `x::WeightedValue`: A WeightedValue with value and precision

# Returns
An uncertain value with:
- value = `value(x)`
- uncertainty = `1 / sqrt(precision(x))`

# Notes
Uncertainty is the standard deviation.

# Example
```julia
using Uncertain, WeightedData
w = WeightedValue(1.0, 100.0)  # value = 1.0, precision = 100.0
u = Uncertain.Value(w)  # Uncertain.Value(1.0, 0.1)
```
"""
(::Type{T})(x::WeightedValue) where {T <: Uncertain.Value} = T(value(x), 1 / sqrt(precision(x)))

end
