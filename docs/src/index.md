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

```@index
```
