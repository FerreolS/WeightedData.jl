```@meta
CurrentModule = WeightedData
```

# WeightedData

`WeightedData.jl` provides weighted numeric containers and likelihood utilities
for uncertainty-aware estimation and model fitting.

## Installation

```julia
using Pkg
Pkg.add("WeightedData")
```

## Why WeightedData?

- Keep values and their precisions together in a single, type-stable container.
- Compute weighted statistics while handling missing or invalid measurements.
- Evaluate likelihoods directly from weighted observations and model predictions.

## Core concepts

- `WeightedValue(value, precision)` stores a scalar observation and its precision.
- `WeightedArray(values, precisions)` stores array-valued observations and per-entry precisions.
- Precision is interpreted as inverse variance, $w = 1 / \sigma^2$.

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

## Common workflows

```julia
using WeightedData

# Weighted means from two observations
a = WeightedValue(1.2, 2.0)
b = WeightedValue(0.8, 1.0)
m = weightedmean(a, b)

# Mark invalid entries before analysis
w = WeightedArray([1.0, NaN, 3.0], [1.0, 1.0, 1.0])
flagbadpix!(w)

# Likelihood of model predictions
obs = WeightedArray([2.0, 1.0], [4.0, 0.5])
pred = [1.8, 1.2]
ℓ = likelihood(obs, pred)
```

## Extensions

`WeightedData.jl` provides optional extensions that are activated automatically
when the corresponding package is loaded:

- `ChainRulesCore` → custom `rrule` methods for `likelihood`
- `Measurements` → conversion between `Measurement` and `WeightedValue`
- `OnlineSampleStatistics` → conversion from online mean/variance statistics to `WeightedValue` and `WeightedArray`
- `RobustModels` → robust `likelihood` and `get_weight` methods
- `Uncertain` → conversion between `Uncertain.Value` and `WeightedValue`

See [API Reference](api.md) for full method docstrings, including extension methods.
