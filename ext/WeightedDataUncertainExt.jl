module WeightedDataUncertainExt
import Uncertain
using WeightedData
import WeightedData: WeightedValue, get_value, get_precision

"""
    WeightedValue(x::Uncertain.Value)

Convert an Uncertain.Value to a WeightedValue, using the uncertainty to determine precision.

# Arguments
- `x::Uncertain.Value`: An Uncertain value with uncertainty

# Returns
A WeightedValue with value and precision derived from the Uncertain value's value and uncertainty.

# Example
```julia
using Uncertain, WeightedData
u = Uncertain.Value(1.0, 0.1)  # value = 1.0, uncertainty = 0.1
w = WeightedValue(u)  # WeightedValue(1.0, 100.0)
```
"""
WeightedData.WeightedValue(x::Uncertain.Value) = WeightedValue(Uncertain.value(x), Uncertain.uncertainty(x)^(-2))

"""
    T(x::WeightedValue) where T <: Uncertain.Value

Convert a WeightedValue to an Uncertain value type, using precision to determine uncertainty.

# Arguments
- `x::WeightedValue`: A WeightedValue with value and precision

# Returns
An Uncertain value with value and uncertainty derived from the WeightedValue's value and precision.

# Example
```julia
using Uncertain, WeightedData
w = WeightedValue(1.0, 100.0)  # value = 1.0, precision = 100.0
u = Uncertain.Value(w)  # Uncertain.Value(1.0, 0.1)
```
"""
(::Type{T})(x::WeightedValue) where {T <: Uncertain.Value} = T(get_value(x), 1 / sqrt(get_precision(x)))

end
