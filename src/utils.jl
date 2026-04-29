"""
    mean(A::WeightedValue, B::Vararg{WeightedValue})

Compute the precision-weighted mean of one or more scalar weighted values.

Equivalent to reducing the inputs with the same precision-weighted averaging
rule used for tuples/iterables.
"""
mean(A::WeightedValue, B::Vararg{WeightedValue}) = mean(tuple(A, B...))
mean(A::WeightedValue) = A

"""
    mean(A::AbstractArray{<:WeightedValue, N}, B::AbstractArray{<:WeightedValue, N}, C...; dims=:) where {N}

Element-wise precision-weighted mean of weighted arrays with matching shape.

- `mean(A, B, ...)` returns an element-wise weighted mean.
- `mean(A, B, ...; dims=:)` reduces along `dims` and across the array list.
"""
function mean(A::AbstractArray{S1, N}, B::AbstractArray{S2, N}, C...; dims = :) where {S1 <: WeightedValue, S2 <: WeightedValue, N}
    all(c -> isa(c, AbstractArray) && eltype(c) <: WeightedValue && ndims(c) == N, C) || throw(ArgumentError("mean: all arrays must contain WeightedValue and have matching dimensionality"))
    stacked = cat(A, B, C...; dims = N + 1)

    if dims isa Colon
        return dropdims(mean(stacked; dims = N + 1), dims = N + 1)
    elseif dims isa Int
        redims = (dims, N + 1)
    elseif dims isa Tuple{Vararg{Int}}
        redims = Tuple(unique((dims..., N + 1)))
    else
        throw(ArgumentError("mean: dims must be :, an Int, or a tuple of Int"))
    end

    return dropdims(mean(stacked; dims = redims), dims = redims)
end

"""
    mean(iterable; dims=:)

Compute a precision-weighted mean from an iterable of `WeightedValue`.

- With `dims = :` (default), returns a scalar `WeightedValue`.
- With `dims` set to one or more dimensions, returns a `WeightedArray`.

`iterable` must be non-empty, have `WeightedValue` elements, and expose an
element type.
"""
mean(iterable::Tuple{Vararg{WeightedValue}}; dims = :) = _weighted_mean(iterable; dims = dims)
mean(iterable::AbstractArray{<:WeightedValue}; dims = :) = _weighted_mean(iterable; dims = dims)

function _weighted_mean(iterable; dims = :)
    length(iterable) == 0 && return throw(ArgumentError("mean: iterable must not be empty"))
    Base.IteratorEltype(iterable) == Base.HasEltype() || throw(ArgumentError("mean: iterable must have eltype"))
    T = TypeUtils.get_precision(first(iterable))

    if dims isa Colon
        out = mapreduce((a, b) -> map(.+, a, b), iterable; init = (zero(T), zero(T))) do el::WeightedValue
            w = get_precision(el)
            get_value(el) * w, w
        end
        invsumw = inv(out[2])
        return WeightedValue(out[1] * invsumw, out[2])
    end

    out = mapreduce((a, b) -> map(.+, a, b), iterable; dims = dims, init = (zero(T), zero(T))) do el::WeightedValue
        w = get_precision(el)
        get_value(el) * w, w
    end
    invsumw = map(x -> inv(x[2]), out)
    return WeightedArray(map((x, y) -> y * x[1], out, invsumw), map(x -> x[2], out))
end
