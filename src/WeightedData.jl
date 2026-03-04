module WeightedData
import ZippedArrays: ZippedArray

import TypeUtils
import StatsAPI: loglikelihood

import Statistics: mean, var, std

export WeightedValue,
    likelihood,
    WeightedArray,
    get_value,
    get_precision,
    loglikelihood,
    get_weights,
    mean, var, std

if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :ScaledL2Loss,
        )
    )
else
    @eval begin
        export ScaledL2Loss
    end
end

include("WeightedValue.jl")
include("WeightedArray.jl")
include("utils.jl")
include("likelihood.jl")

@deprecate value(args...; kwargs...) get_value(args...; kwargs...)
@deprecate precision(args...; kwargs...) get_precision(args...; kwargs...)

end
