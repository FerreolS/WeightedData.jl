# WeightedData

[![License][license-img]][license-url] [![Build Status][github-ci-img]][github-ci-url] [![Coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url]

A Julia package to manipulate data weighted by their precision and compute likelihood functions.

## Usage

```julia
using WeightedData

# Create weighted points
x = WeightedValue(1.0, 0.5)  # value 1.0 with precision 0.5
y = WeightedValue(2.0, 0.2)  # value 2.0 with precision 0.2

# weightedmean points (weighted average)
z = weightedmean(x, y)  

# build an array of weighted points
data = WeightedArray([1.0, 1.0], [2.0, 0.5])

# Compute likelihood
model = [1.0, 1.5]
l = likelihood(data, model)  # Default Gaussian likelihood

# Compute derivative with autodifferentiation
f(x) = likelihood(data, x)
using Zygote
lkl, grad = Zygote.withgradient(f, model)


# Robust likelihood
using RobustModels
l_robust = likelihood(data, model, loss=HuberLoss())
```

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[github-ci-img]: https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml?query=branch%3Amaster
[codecov-img]: http://codecov.io/github/FerreolS/WeightedData.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/FerreolS/WeightedData.jl?branch=master
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl