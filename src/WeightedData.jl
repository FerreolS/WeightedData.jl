module WeightedData
import ZippedArrays: ZippedArray

import TypeUtils
import StatsAPI: loglikelihood

import Statistics: mean, var, std

export WeightedValue,
    likelihood,
    WeightedArray,
    value,
    precision,
    loglikelihood,
    get_weights,
    mean, var, std

if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :ScaledL2Loss,
            :flagbaddata,
            :flagbaddata!,
        )
    )
else
    @eval begin
        export ScaledL2Loss, flagbaddata, flagbaddata!
    end
end

include("WeightedValue.jl")
include("WeightedArray.jl")
include("utils.jl")
include("likelihood.jl")

@deprecate get_value(args...; kwargs...) value(args...; kwargs...)
@deprecate get_precision(args...; kwargs...) precision(args...; kwargs...)

end
