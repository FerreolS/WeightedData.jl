## Array of WeightedPoint
using ZippedArrays

get_data(x::AbstractArray{<:WeightedPoint}) = map(x -> x.data, x)
get_precision(x::AbstractArray{<:WeightedPoint}) = map(x -> x.precision, x)
WeightedPoint(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = map((a, b) -> WeightedPoint(a, b), A, B)


WeightedArray{T,N} = ZippedArray{WeightedPoint{T},N,2,I,Tuple{A,B}} where {A<:AbstractArray{T,N},B<:AbstractArray{T,N},I}
WeightedArray(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = ZippedArray{WeightedPoint{T}}(A, B)
WeightedArray(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = ZippedArray{WeightedPoint{T1}}(A, T1.(B))


WeightedArray(x::AbstractArray{<:WeightedPoint}) = WeightedArray(get_data(x), get_precision(x))
WeightedArray(x::WeightedArray) = x

function WeightedArray(x::AbstractArray{<:Union{T,Missing}}) where {T<:Real}
    m = .!ismissing.(x) .&& .!isnan.(x)
    return WeightedArray(ifelse.(m, x, T(0)), m)
end
WeightedArray(x::AbstractArray{Missing}) = WeightedArray(zeros(size(x)), zeros(size(x)))
WeightedArray(x::AbstractArray{T}) where {T<:Real} = WeightedArray(x, ones(size(x)))

Base.view(A::WeightedArray, I...) = WeightedArray(view(get_data(A), I...), view(get_precision(A), I...))

get_data(x::WeightedArray) = x.args[1]
get_precision(x::WeightedArray) = x.args[2]

Base.propertynames(::WeightedArray) = (:data, :precision)
function Base.getproperty(A::WeightedArray, s::Symbol)
    if s == :data
        return get_data(A)
    elseif s == :precision
        return get_precision(A)
    else
        getfield(A, s)
    end
end

function flagbadpix(data::AbstractArray{WeightedPoint{T},N}, badpix::Union{Array{Bool,N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return @inbounds map((d, flag) -> ifelse(flag, WeightedPoint(T(0), T(0)), d), data, badpix)
end

function flagbadpix!(data::WeightedArray{T1,N}, badpix::Union{AbstractArray{Bool,N},BitArray{N}}) where {T1,N}
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