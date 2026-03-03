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

```@repl
using WeightedData
import Statistics: mean, var, std
import StatsAPI: loglikelihood

x = WeightedValue(1.0, 0.5)
y = WeightedValue(2.0, 0.2)
z = mean(x, y)
vx = var(x)
sx = std(x)

data = WeightedArray([1.0, 1.0], [2.0, 0.5])
model = [1.0, 1.5]
ll = loglikelihood(data, model)
```

## Common workflows

```@repl
using WeightedData
import Statistics: mean, var, std
import WeightedData: flagbaddata!

# Weighted means from two observations
a = WeightedValue(1.2, 2.0)
b = WeightedValue(0.8, 1.0)
m = mean(a, b)
va = var(a)

# Global weighted mean over a weighted array
wa = WeightedArray([1.0, 2.0, 3.0], [1.0, 1.0, 0.5])
mg = mean(wa)
vg = var(wa)
sg = std(wa)

# Mark invalid entries before analysis
w = WeightedArray([1.0, NaN, 3.0], [1.0, 1.0, 1.0])
flagbaddata!(w)

# Likelihood of model predictions
obs = WeightedArray([2.0, 1.0], [4.0, 0.5])
pred = [1.8, 1.2]
ℓ = loglikelihood(obs, pred)
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

### What is supported

- `WeightedArray` storage on GPU arrays (for example `CuArray`) for values and precisions.
- `loglikelihood(loss, data, model)` where `data` is GPU-backed and `model` has matching shape.
- `flagbaddata` and `flagbaddata!` on GPU-backed weighted arrays.

### Example (CUDA)

```@example
using WeightedData
using CUDA
using RobustModels
import StatsAPI: loglikelihood

values = CUDA.ones(Float32, 1024)
precisions = CUDA.fill(Float32(2), 1024)
data = WeightedArray(values, precisions)
model = CUDA.fill(Float32(0.9), 1024)

# Gaussian (default) negative log-likelihood
ℓ1 = loglikelihood(data, model)

# Robust negative log-likelihood (requires RobustModels extension)
ℓ2 = loglikelihood(data, model, loss=HuberLoss())

# Flag invalid points from a Boolean mask
badmask = falses(1024)
badmask[10] = true
flagbaddata!(data, badmask)
```

Notes:

- The `data` and `model` arrays must have identical shapes.
- A CPU `badmask` (`Array` or `BitArray`) is accepted and adapted to the GPU backend.
- Backend choice is delegated to your GPU array package (for example `CUDA.jl`, `AMDGPU.jl`, `oneAPI.jl`).
