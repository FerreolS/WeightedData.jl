module WeightedDataMeasurementsExt
import Measurements
import WeightedData: WeightedValue, get_value, get_precision

"""
    WeightedValue(x::Measurements.Measurement)

Convert a `Measurements.Measurement` to `WeightedValue`.

# Arguments
- `x::Measurements.Measurement`: value with uncertainty

# Returns
A `WeightedValue` with:
- value = `Measurements.value(x)`
- precision = `Measurements.uncertainty(x)^(-2)`

# Notes
Precision is the inverse variance.

# Example
```julia
using Measurements, WeightedData
m = 1.0 ± 0.1  # Measurement with value 1.0 and uncertainty 0.1
w = WeightedValue(m)  # WeightedValue(1.0, 100.0)
```
"""
WeightedValue(x::Measurements.Measurement) = WeightedValue(Measurements.value(x), Measurements.uncertainty(x)^(-2))

"""
    Measurement(x::WeightedValue)

Convert `WeightedValue` to `Measurements.Measurement`.

# Arguments
- `x::WeightedValue`: A WeightedValue with value and precision

# Returns
A `Measurements.Measurement` with:
- value = `get_value(x)`
- uncertainty = `1 / sqrt(get_precision(x))`

# Notes
Uncertainty is the standard deviation.

# Example
```julia
using Measurements, WeightedData
w = WeightedValue(1.0, 100.0)  # value = 1.0, precision = 100.0
m = Measurement(w)  # 1.0 ± 0.1
```
"""
Measurements.Measurement(x::WeightedValue) = Measurements.measurement(get_value(x), 1 / sqrt(get_precision(x)))

end
