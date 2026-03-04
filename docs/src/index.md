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

## Common workflows

```@repl

using WeightedData

using WeightedData: filterbaddata!, get_value, get_precision, get_weights

# Define arrays of values and precisions
values = [ 	1.0 missing π
            0.1 10 		NaN]

precision = [0 	missing 5
            0.1 10 		3.]

# Create a WeightedArray
data = WeightedArray(values, precision)

# Missing and non-numeric values are ignored by setting their precision to zero
get_precision(data)

# Elements are `WeightedValue`s, displayed with their standard deviation
data[2]

# Weighted mean along a given dimension
mean(data, dims=1)

model = [1.0 2.0 3.
        3 	2. 	1. ]


# Compute the negative log-likelihood for a given model
l = loglikelihood(data, model)  # Default Gaussian negative log-likelihood

# Compute the derivative with automatic differentiation
f(x) = loglikelihood(data, x)

using Zygote

lkl, grad = Zygote.withgradient(f, model)

# Use a robust loss from the RobustModels package
using RobustModels

l_robust = loglikelihood(data, model, loss=HuberLoss())

# Robust weights computed by the model (outlier weights are lower than weights of valid data points)
get_weights(data, model;loss=HuberLoss())
```

## Extensions

`WeightedData.jl` provides optional extensions that are activated automatically
when the corresponding package is loaded:

- `RobustModels` → robust `loglikelihood` taking any Losses defined in `RobustModels`
- `OnlineSampleStatistics` → extract sample mean and sample variance of a series of observation to build a WeightedData (`WeightedValue` or `WeightedArray`)
- `ChainRulesCore` → custom `rrule` methods for `loglikelihood`
- `Measurements` → conversion between `Measurement` and `WeightedValue`
- `Uncertain` → conversion between `Uncertain.Value` and `WeightedValue`

See [API Reference](api.md) for full method docstrings, including extension methods.

## GPU support

`WeightedData.jl` supports GPU-backed weighted arrays through the
`WeightedDataGPUArraysExt` extension, activated automatically when
`GPUArrays.jl` is loaded.

### Example (CUDA)

```julia
using WeightedData
using CUDA
using RobustModels

values = CUDA.ones(Float32, 1024)
precisions = CUDA.fill(Float32(2), 1024)
data = WeightedArray(values, precisions)
model = CUDA.fill(Float32(0.9), 1024)

# Gaussian (default) negative log-likelihood
ℓ2 = loglikelihood(data, model)

# Robust negative log-likelihood (requires RobustModels extension)
ℓh = loglikelihood(data, model, loss=HuberLoss())

```

Notes:

- Backend choice is delegated to your GPU array package (for example `CUDA.jl`, `AMDGPU.jl`, `oneAPI.jl`).
