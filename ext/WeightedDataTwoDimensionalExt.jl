module WeightedDataTwoDimensionalExt
import TwoDimensional: BoundingBox
import WeightedData: value, precision, WeightedArray


function Base.view(A::WeightedArray, I::BoundingBox{<:Integer})
    return WeightedArray(view(value(A), I), view(precision(A), I))
end
end
