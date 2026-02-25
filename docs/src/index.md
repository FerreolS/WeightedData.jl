```@meta
CurrentModule = WeightedData
```

# WeightedData

`WeightedData.jl` provides weighted numeric containers and likelihood utilities.

## Quick start

```julia
using WeightedData

x = WeightedValue(1.0, 0.5)
y = WeightedValue(2.0, 0.2)
z = weightedmean(x, y)

data = WeightedArray([1.0, 1.0], [2.0, 0.5])
model = [1.0, 1.5]
ll = likelihood(data, model)
```

## Extensions

`WeightedData.jl` provides optional extensions that are activated automatically
when the corresponding package is loaded:

- `ChainRulesCore` → custom `rrule` methods for `likelihood`
- `Measurements` → conversion between `Measurement` and `WeightedValue`
- `RobustModels` → robust `likelihood` and `get_weight` methods
- `Uncertain` → conversion between `Uncertain.Value` and `WeightedValue`

See the API page for extension method docstrings.
