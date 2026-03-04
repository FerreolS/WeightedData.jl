```@meta
CurrentModule = WeightedData
```

# API

## Types

```@docs
WeightedValue
WeightedArray
WeightedArray(::AbstractArray{<:Union{Missing, Real}, N}, ::AbstractArray{<:Union{Missing, Real}, N}) where {N}
```

## Exported Functions

```@docs
loglikelihood
mean
var
std
```

## Public Functions

```@docs
get_value
get_precision
get_weights
filterbaddata!
ScaledL2Loss
```

## Extension Functions

### Measurements

```@autodocs
Modules = Main.DOC_EXT_MEASUREMENTS
Private = true
```

### OnlineSampleStatistics

```@autodocs
Modules = Main.DOC_EXT_ONLINESAMPLESTATISTICS
Private = true
```

### RobustModels

```@autodocs
Modules = Main.DOC_EXT_ROBUSTMODELS
Private = true
```

### Uncertain

```@autodocs
Modules = Main.DOC_EXT_UNCERTAIN
Private = true
```
