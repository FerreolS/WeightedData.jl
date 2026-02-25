module WeightedDataOnlineSampleStatisticsExt

import OnlineSampleStatistics: IndependentStatistic, UnivariateStatistic
import OnlineSampleStatistics: mean, var
import WeightedData: WeightedValue, WeightedArray

"""
    WeightedValue(x::UnivariateStatistic{T, K}) where {T, K}

Convert a univariate online statistic to a `WeightedValue`.

The returned value is `mean(x)` and the returned precision is `inv(var(x))`.
At least two moments (`K ≥ 2`) are required so that the variance is available.
"""
function WeightedValue(x::UnivariateStatistic{T, K}) where {T, K}
    2 ≤ K || throw(ArgumentError("Variance (K=2) is mandatory to build a WeightedValue from a UnivariateStatistic"))
    return WeightedValue{T}(mean(x), inv(var(x)))
end

"""
    WeightedArray(x::IndependentStatistic{T, N, K, W}) where {T, N, K, W}

Convert an independent online statistic array to a `WeightedArray`.

The returned values are `mean(x)` and element-wise precisions `inv.(var(x))`.
At least two moments (`K ≥ 2`) are required so that the variance is available.
"""
function WeightedArray(x::IndependentStatistic{T, N, K, W}) where {T, N, K, W}
    2 ≤ K || throw(ArgumentError("Variance (K=2) is mandatory to build a WeightedArray from an IndependentStatistic"))
    return WeightedArray(mean(x), inv.(var(x)))
end

end
