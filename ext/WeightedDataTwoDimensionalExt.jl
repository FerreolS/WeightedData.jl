module WeightedDataTwoDimensionalExt
using TwoDimensional
import WeightedData: get_data, get_precision, WeightedArray


function Base.view(A::WeightedArray, I::BoundingBox{<:Integer})
    WeightedArray(view(get_data(A), I), view(get_precision(A), I))
end
end
