## Array of WeightedPoint

get_val(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x -> x.val, x)
get_precision(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x -> x.precision, x)

WeightedPoint(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = map((a, b) -> WeightedPoint(a, b), A, B)

function flagbadpix(data::AbstractArray{WeightedPoint{T},N}, badpix::Union{Array{Bool,N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(A) != size(badpix)")
    return map((d, flag) -> ifelse(flag, WeightedPoint(T(0), T(0)), d), data, badpix)
end
