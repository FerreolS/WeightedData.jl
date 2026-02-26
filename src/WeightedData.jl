module WeightedData
import ZippedArrays: ZippedArray

import TypeUtils


export WeightedValue,
    likelihood,
    WeightedArray,
    get_value,
    get_precision,
    get_weight

if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :weightedmean,
            :ScaledL2Loss,
            :flagbaddata,
            :flagbaddata!,
        )
    )
else
    @eval begin
        export weightedmean, ScaledL2Loss, flagbaddata, flagbaddata!
    end
end

include("WeightedValue.jl")
include("WeightedArray.jl")
include("weightedmean.jl")
include("likelihood.jl")

end
