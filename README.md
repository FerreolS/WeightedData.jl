# WeightedData

[![Build Status](https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/FerreolS/WeightedData.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia package to manipulate data weighted by their precision and compute likelihood functions.

## Usage

```julia
using WeightedData

# Create weighted points
x = WeightedPoint(1.0, 0.5)  # value 1.0 with precision 0.5
y = WeightedPoint(2.0, 0.2)  # value 2.0 with precision 0.2

# Combine points (weighted average)
z = combine(x, y)  

# build an array of weighted points
data = WeightedPoint([1.0, 1.0], [2.0, 0.5])

# Compute likelihood
model = [1.0, 1.5]
l = likelihood(data, model)  # Default Gaussian likelihood

# Robust likelihood
l_robust = likelihood(data, model, likelihoodfunc=robustlikelihood(3))
```
