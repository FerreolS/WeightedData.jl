module WeightedDataTwoDimensionalExt
using TwoDimensional
import WeightedData: get_value, get_precision, WeightedArray


function Base.view(A::WeightedArray, I::BoundingBox{<:Integer})
    WeightedArray(view(get_value(A), I), view(get_precision(A), I))
end
end
