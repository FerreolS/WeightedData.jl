module WeightedDataRobustModelsGPUArraysExt

import WeightedData: loglikelihood, WeightedValue, get_value, get_precision
import RobustModels: LossFunction, rho
import ZippedArrays: ZippedArray
import GPUArrays: AnyGPUArray
import ChainRulesCore

WeightedArrayGPU{T, N}  = ZippedArray{WeightedValue{T},N,2,I,Tuple{A, A}} where {T,N,I,A <: AnyGPUArray{T,N}}

function loglikelihood(loss::LossFunction, data::WeightedArrayGPU{T1, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
    size(data) == size(model) || error("loglikelihood : size(A) != size(model)")
    g = Base.Fix1(rho,loss)
    r = sqrt.(get_precision(data)) .* (model .- get_value(data))
    return sum(g.(r))
    #return mapreduce((value, precision, model) -> g(sqrt(precision) * (model - value)), +, get_value(data),get_precision(data) , model; init = zero(T2))
end

end