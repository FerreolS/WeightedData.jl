module WeightedDataMeasurementsExt
using Measurements
using WeightedData
import Measurements: Measurement, measurement, value, uncertainty
import WeightedData: WeightedValue, get_value, get_precision

"""
    WeightedValue(x::Measurements.Measurement)

Convert a Measurement to a WeightedValue, using the measurement uncertainty to determine precision.

# Arguments
- `x::Measurements.Measurement`: A value with measurement uncertainty

# Returns
A WeightedValue with value and precision derived from the measurement's value and uncertainty.

# Details
The precision is computed as `uncertainty^(-2)`, i.e., the inverse square of the uncertainty.

# Example
```julia
using Measurements, WeightedData
m = 1.0 ± 0.1  # Measurement with value 1.0 and uncertainty 0.1
w = WeightedValue(m)  # WeightedValue(1.0, 100.0)
```
"""
WeightedData.WeightedValue(x::Measurements.Measurement) = WeightedValue(Measurements.value(x), Measurements.uncertainty(x)^(-2))

"""
    Measurement(x::WeightedValue)

Convert a WeightedValue to a Measurement, using precision to determine uncertainty.

# Arguments
- `x::WeightedValue`: A WeightedValue with value and precision

# Returns
A Measurement with value and uncertainty derived from the WeightedValue's value and precision.

# Details
The uncertainty is computed as `1 / sqrt(precision)`.

# Example
```julia
using Measurements, WeightedData
w = WeightedValue(1.0, 100.0)  # value = 1.0, precision = 100.0
m = Measurement(w)  # 1.0 ± 0.1
```
"""
Measurements.Measurement(x::WeightedValue) = Measurements.measurement(get_value(x), 1 / sqrt(get_precision(x)))
end
