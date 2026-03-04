```@meta
CurrentModule = WeightedData
```

# API

## Types

```@docs
WeightedValue
WeightedArray
```

## Exported Functions

```@docs
loglikelihood
get_weights
mean
var
std
```

## Public Functions

```@docs
get_value
get_precision
filterbaddata!
ScaledL2Loss
```

## Extension Functions

### Measurements

```@autodocs
Modules = Main.DOC_EXT_MEASUREMENTS
Order = [:function]
Private = true
```

### OnlineSampleStatistics

```@autodocs
Modules = Main.DOC_EXT_ONLINESAMPLESTATISTICS
Order = [:function]
Private = true
```

### RobustModels

```@autodocs
Modules = Main.DOC_EXT_ROBUSTMODELS
Order = [:function]
Private = true
```

### Uncertain

```@autodocs
Modules = Main.DOC_EXT_UNCERTAIN
Order = [:function]
Private = true
```
