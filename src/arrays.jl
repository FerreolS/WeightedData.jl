## Array of WeightedPoint
using ZippedArrays

get_val(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x -> x.val, x)
get_precision(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x -> x.precision, x)
WeightedPoint(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = map((a, b) -> WeightedPoint(a, b), A, B)

function flagbadpix(data::AbstractArray{WeightedPoint{T},N}, badpix::Union{Array{Bool,N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return map((d, flag) -> ifelse(flag, WeightedPoint(T(0), T(0)), d), data, badpix)
end

WeightedArray{T,N} = ZippedArray{WeightedPoint{T},N,2,I,Tuple{A,A}} where {T,N,A<:AbstractArray{T,N},I}
WeightedArray(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = ZippedArray{WeightedPoint{T1}}(A, T1.(B))

get_val(x::WeightedArray) = x.args[1]
get_precision(x::WeightedArray) = x.args[2]

function flagbadpix!(data::WeightedArray{T1,N,B,I}, badpix::Union{AbstractArray{Bool,N},BitArray{N}}) where {B,T1,N,I}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return data[badpix] .= WeightedPoint{T1}(T1(0), T1(0))
end
