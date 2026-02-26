"""
    weightedmean(A::WeightedValue, B::Vararg{<:WeightedValue})

Compute the precision-weighted mean of one or more scalar weighted values.

Equivalent to reducing the inputs with the same precision-weighted averaging
rule used for tuples/iterables.
"""
weightedmean(A::WeightedValue, B::Vararg{<:WeightedValue}) = weightedmean(tuple(A, B...))

"""
    weightedmean(A::AbstractArray{<:WeightedValue, N}, B::Vararg{AbstractArray{<:WeightedValue, N}}) where {N}

Element-wise precision-weighted mean of weighted arrays with matching shape.

- `weightedmean(A)` returns the global weighted mean of all entries.
- `weightedmean(A, B, ...)` returns an element-wise weighted mean.
"""
function weightedmean(A::AbstractArray{<:WeightedValue, N}, B::Vararg{AbstractArray{<:WeightedValue, N}}) where {N}
    isempty(B) && return weightedmean(A; dims = :)
    return dropdims(weightedmean(cat(A, B...; dims = N + 1); dims = N + 1), dims = N + 1)
end

"""
    weightedmean(iterable; dims=:)

Compute a precision-weighted mean from an iterable of `WeightedValue`.

- With `dims = :` (default), returns a scalar `WeightedValue`.
- With `dims` set to one or more dimensions, returns a `WeightedArray`.

`iterable` must be non-empty, have `WeightedValue` elements, and expose an
element type.
"""
function weightedmean(iterable; dims = :)
    length(iterable) == 0 && return throw(ArgumentError("weightedmean: iterable must not be empty"))
    first(iterable) isa WeightedValue || throw(ArgumentError("weightedmean: iterable must contain WeightedValue"))
    Base.IteratorEltype(iterable) == Base.HasEltype() || throw(ArgumentError("weightedmean: iterable must have eltype"))
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
