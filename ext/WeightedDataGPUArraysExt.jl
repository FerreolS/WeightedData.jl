module WeightedDataGPUArraysExt

import WeightedData:WeightedArray, WeightedValue, loglikelihood,flagbaddata, flagbaddata!, value, precision
import GPUArrays: adapt, AnyGPUArray, @allowscalar
import ZippedArrays: ZippedArray
import Base: Base, show


"""
    WeightedArrayGPU{T, N}

Alias for a `WeightedArray`-like zipped container whose value and precision
storage live on GPU-compatible arrays (`AnyGPUArray`).
"""
WeightedArrayGPU{T, N}  = ZippedArray{WeightedValue{T},N,2,I,Tuple{A, A}} where {T,N,I,A <: AnyGPUArray{T,N}}
# typeinfo aware
# implements: show(io::IO, ::MIME"text/plain", X::AbstractArray)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, X::WeightedArrayGPU) 
    if isempty(X) && (get(io, :compact, false)::Bool || X isa Vector)
        return show(io, X)
    end
    # 1) show summary before setting :compact
    summary(io, X)
    isempty(X) && return
    print(io, ":")
    Base.show_circular(io, X) && return

    # 2) compute new         IOContext
    if !haskey(io, :compact) && length(axes(X, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false)::Bool && eltype(X) === Method
        # override usual show method for Vector{Method}: don't abbreviate long lists
        io = IOContext(io, :limit => false)
    end

    if get(io, :limit, false)::Bool && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 3) update typeinfo
    #
    # it must come after printing the summary, which can exploit :typeinfo itself
    # (e.g. views)
    # we assume this function is always called from top-level, i.e. that it's not nested
    # within another "show" method; hence we always print the summary, without
    # checking for current :typeinfo (this could be changed in the future)
    io = IOContext(io, :typeinfo => eltype(X))

    # 4) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    @allowscalar Base.print_array(recur_io, X)
   # Base.print_array(recur_io, adapt(Array, X))
end


"""
    loglikelihood(loss, data::WeightedArrayGPU, model::AbstractArray)

Compute `loglikelihood` for GPU-backed weighted arrays using the provided loss
kernel and a reduction over `(value, precision, model)` tuples.
"""
function loglikelihood(loss, data::WeightedArrayGPU{T1, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
        size(data) == size(model) || error("loglikelihood : size(A) != size(model)")
        return mapreduce(loss, +, data.args..., model)
end

function flagbaddata(data::WeightedArrayGPU{T, N}, badmask::Union{Array{Bool, N}, BitArray{N}}) where {T, N}
    size(data) == size(badmask) || error("flagbaddata! : size(data) != size(badmask)")
    badmask = adapt(get_wrapper(data), badmask)
    newprecision = map((v, p, flag) -> ifelse(flag, zero(typeof(p)), p), value(data), precision(data), badmask)
    return WeightedArray(value(data), newprecision)
end

function flagbaddata(data::WeightedArrayGPU{T, N}, badmask::AbstractArray{Bool, N}) where {T, N}
    size(data) == size(badmask) || error("flagbaddata! : size(data) != size(badmask)")
    newprecision = map((v, p, flag) -> ifelse(flag, zero(typeof(p)), p), value(data), precision(data), badmask)
    return WeightedArray(value(data), newprecision)
end

function flagbaddata!(data::WeightedArrayGPU{T, N}, badmask::Union{Array{Bool, N}, BitArray{N}}) where {T, N}
    size(data) == size(badmask) || error("flagbaddata! : size(data) != size(badmask)")
    badmask = adapt(get_wrapper(data), badmask)
    map!((v, p, flag) -> ifelse(flag, zero(typeof(p)), p), precision(data), value(data), precision(data), badmask)
    return data
end

function flagbaddata!(data::WeightedArrayGPU{T, N}, badmask::AbstractArray{Bool, N}) where {T, N}
    size(data) == size(badmask) || error("flagbaddata! : size(data) != size(badmask)")
    map!((v, p, flag) -> ifelse(flag, zero(typeof(p)), p), precision(data), value(data), precision(data), badmask)
    return data
end

get_wrapper(x::ZippedArray{T,N,L,I,<:NTuple{L,S}} ) where {T,N,L,I,S <: AnyGPUArray}  = S.name.wrapper

end