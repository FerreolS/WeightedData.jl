## Array of Measurement
using ZippedArrays

get_data(x::AbstractArray{<:Measurement}) = map(x -> x.val, x)
get_precision(x::AbstractArray{<:Measurement}) = map(x -> inv(x.err)^2, x)

get_data(x::Measurement) = x.val
get_precision(x::Measurement) = inv(x.err)^2

const WeightedArray{T,N} = ZippedArray{Measurement{T},N,2,I,Tuple{A,B}} where {A<:AbstractArray{T,N},B<:AbstractArray{T,N},I}

ZippedArrays.build(::Type{Measurement{T}}, (val, weight)::Tuple) where {T} = measurement(val, sqrt(inv(weight)))


WeightedArray(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = ZippedArray{Measurement{T}}(A, B)
WeightedArray(A::AbstractArray{T1,N}, B::AbstractArray{T2,N}) where {T1,T2,N} = ZippedArray{Measurement{T1}}(A, T1.(B))


WeightedArray(x::AbstractArray{<:Measurement}) = WeightedArray(get_data(x), get_precision(x))
WeightedArray(x::WeightedArray) = x

function WeightedArray(x::AbstractArray{<:Union{T,Missing}}) where {T<:Real}
    m = .!ismissing.(x) .&& .!isnan.(x)
    return WeightedArray(ifelse.(m, x, zero(T)), m)
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

function flagbadpix(data::AbstractArray{Measurement{T},N}, badpix::Union{Array{Bool,N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    return @inbounds map((d, flag) -> ifelse(flag, measurement(zero(T), T(Inf)), d), data, badpix)
end

function flagbadpix!(data::WeightedArray{T1,N}, badpix::Union{AbstractArray{Bool,N},BitArray{N}}) where {T1,N}
    size(data) == size(badpix) || error("flagbadpix! : size(data) != size(badpix)")
    get_data(data)[badpix] .= zero(T1)
    get_precision(data)[badpix] .= T1(0.0)
end


function Measurements.weightedmean(iterable::Union{Vector{<:WeightedArray},NTuple{N,<:WeightedArray}}) where {N}
    length(iterable) == 1 && return iterable[1]
    reduce(weightedmean, iterable)
end

function Measurements.weightedmean(A::WeightedArray{T,N}, B::WeightedArray{T,N}) where {N,T}
    size(A) == size(B) || error("weightedmean : size(A) != size(B)")
    ABprecision = get_precision(A) .+ get_precision(B)
    WeightedArray(
        (get_data(A) .* get_precision(A) .+ get_data(B) .* get_precision(B)) ./ ABprecision,
        ABprecision
    )
end