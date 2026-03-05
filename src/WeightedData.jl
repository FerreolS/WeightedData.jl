module WeightedData
import ZippedArrays: ZippedArray

import TypeUtils
import StatsAPI: loglikelihood

import Statistics: mean, var, std

export WeightedValue,
    WeightedArray,
    loglikelihood,
    mean, var, std

if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :ScaledL2Loss,
            :filterbaddata!,
            :get_value,
            :get_precision,
            :get_weights
        )
    )
else
    @eval begin
        export ScaledL2Loss,
            filterbaddata!,
            get_value,
            get_precision,
            get_weights
    end
end

include("WeightedValue.jl")
include("WeightedArray.jl")
include("utils.jl")
include("likelihood.jl")

@deprecate value(args...; kwargs...) get_value(args...; kwargs...)
@deprecate precision(args...; kwargs...) get_precision(args...; kwargs...)

import Adapt: adapt_structure, adapt
"""
    adapt_structure(to, wd::WeightedArray)

Adapt a `WeightedArray` structure to a target backend using `Adapt.jl`.

Both value and precision arrays are adapted consistently and wrapped back into
`WeightedArray`.
"""
adapt_structure(to, wd::WeightedArray) =
    _WeightedArray(adapt(to, get_value(wd)), adapt(to, get_precision(wd)))

    
"""
    oncpu(::AbstractArray) -> Bool

Trait to determine whether an array is stored on the CPU.

Returns `true` if the array is located on CPU memory, `false` otherwise.

# Arguments
- `::AbstractArray`: An abstract array to check.

# Returns
- `Bool`: `true` if the array is on CPU, `false` if on other devices (e.g., GPU).
"""
oncpu(::AbstractArray) = true
    
end
