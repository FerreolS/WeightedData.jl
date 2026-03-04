# WeightedData

[![License][license-img]][license-url] [![Build Status][github-ci-img]][github-ci-url] [![Coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] [![Docs][docs-img]][docs-url]

A Julia package for working with data weighted by precision. It provides containers that associate values with their precision (inverse variance), along with likelihood utilities for uncertainty-aware estimation and model fitting.

It mainly provides two data structures:
- `WeightedArray{T, N} <: AbstractArray`, which stores array-valued observations and per-entry precisions of type `T`.
- `WeightedValue{T}`, the element type of `WeightedArray`, which stores a scalar observation and its precision of type `T`.


## Usage

```julia-repl
julia> using WeightedData

julia> using WeightedData: filterbaddata!, get_value, get_precision

# Define arrays of values and precisions
julia> values = [ 	1.0 missing π
                	0.1 10 		NaN]
2×3 Matrix{Union{Missing, Float64}}:
 1.0    missing    3.14159
 0.1  10.0       NaN

julia> precision = [ 0 	missing 5
                   	0.1 10 		3.]
2×3 Matrix{Union{Missing, Float64}}:
 0.0    missing  5.0
 0.1  10.0       3.0

# Create a WeightedArray
julia> data = WeightedArray(values, precision)
2×3 WeightedArray{Float64, 2} (alias of ZippedArrays.ZippedMatrix{WeightedValue{Float64}, 2, true, Tuple{Matrix{Float64}, Matrix{Float64}}})::
 1.0 ± Inf    0.0 ± Inf  3.14159 ± 0.45
 0.1 ± 3.2  10.0 ± 0.32       0.0 ± Inf

# Missing and non-numeric values are ignored by setting their precision to zero
julia> get_precision(data)
2×3 Matrix{Float64}:
 0.0   0.0  5.0
 0.1  10.0  0.0

# Elements are `WeightedValue`s, displayed with their standard deviation
julia> data[2]
WeightedValue{Float64}
0.1 ± 3.2

# Weighted mean along a given dimension
julia> mean(data, dims=1)
1×3 WeightedArray{Float64, 2} (alias of ZippedArrays.ZippedMatrix{WeightedValue{Float64}, 2, true, Tuple{Matrix{Float64}, Matrix{Float64}}})::
 0.1 ± 3.2  10.0 ± 0.32  3.14159 ± 0.45

julia> model = [1.0 2.0 3.
               	3 	2. 	1. ]
2×3 Matrix{Float64}:
 1.0  2.0  3.0
 3.0  2.0  1.0

# Compute the negative log-likelihood for a given model
julia> l = loglikelihood(data, model)  # Default Gaussian negative log-likelihood
320.47062119887653

# Compute the derivative with automatic differentiation
julia> f(x) = loglikelihood(data, x)
f (generic function with 1 method)

julia> using Zygote

julia> lkl, grad = Zygote.withgradient(f, model)
(val = 320.47062119887653, grad = ([0.0 0.0 -0.7079632679489656; 0.29 -80.0 0.0],))

# Use a robust loss from the RobustModels package
julia> using RobustModels

julia> l_robust = loglikelihood(data, model, loss=HuberLoss())
18.56923830366538

# Robust weights computed by the model (outlier weights are lower than weights of valid data points)
julia> get_weights(data, model;loss=HuberLoss())
2×3 Matrix{Float64}:
 1.0  1.0        1.0
 1.0  0.0531658  1.0
```

## GPU support

GPU support is provided through an automatic extension (`WeightedDataGPUArraysExt`) when
`GPUArrays.jl` is loaded.



[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[github-ci-img]: https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml?query=branch%3Amaster
[codecov-img]: http://codecov.io/github/FerreolS/WeightedData.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/FerreolS/WeightedData.jl?branch=master
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://ferreols.github.io/WeightedData.jl/dev/
