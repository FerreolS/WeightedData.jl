## Array of WeightedPoint
using ZippedArrays

get_data(x::AbstractArray{<:WeightedPoint}) = map(x -> x.data, x)
get_precision(x::AbstractArray{<:WeightedPoint})  = map(x -> x.precision, x)
WeightedPoint(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = map((a, b) -> WeightedPoint(a, b), A, B)

function flagbadpix(data::AbstractArray{WeightedPoint{T},N}, badpix::Union{Array{Bool,N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return @inbounds map((d, flag) -> ifelse(flag, WeightedPoint(T(0), T(0)), d), data, badpix)
end

WeightedArray{T,N} = ZippedArray{WeightedPoint{T},N,2,I,Tuple{A,A}} where {A<:AbstractArray{T,N},I}
WeightedArray(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = ZippedArray{WeightedPoint{T1}}(A, T1.(B))

function Base.view(A::WeightedArray, I...)
    WeightedArray(view(A.data, I...), view(A.precision, I...))
end

get_data(x::WeightedArray) = x.args[1]
get_precision(x::WeightedArray) = x.args[2]

function flagbadpix!(data::WeightedArray{T1,N,B,I}, badpix::Union{AbstractArray{Bool,N},BitArray{N}}) where {B,T1,N,I}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return data[badpix] .= WeightedPoint{T1}(T1(0), T1(0))
end

function combine(A::AbstractArray{WeightedPoint{T}}; dims=Colon()) where {T}
    if dims == Colon()
        return reduce(combine, A)
    end
    return mapslices(combine, A; dims=dims)
end

function combine(A::AbstractArray{WeightedPoint{T1}}, B::AbstractArray{WeightedPoint{T2}}) where {T1,T2}
    T = promote_type(T1, T2)
    dataA = get_data(A)
    dataB = get_data(B)
    precisionA = get_precision(A)
    precisionB = get_precision(B)
    precision = T.(precisionA .+ precisionB)
    return WeightedArray(T.(dataA .* precisionA .+ dataB .* precisionB) ./ precision, precision)
end
