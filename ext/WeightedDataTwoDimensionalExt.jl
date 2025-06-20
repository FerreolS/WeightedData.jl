module WeightedDataTwoDimensionalExt
using TwoDimensional
import WeightedData: get_data, get_precision, WeightedArray, weightedarray


function Base.view(A::WeightedArray, I::BoundingBox{<:Integer})
    weightedarray(view(get_data(A), I), view(get_precision(A), I))
end
end
