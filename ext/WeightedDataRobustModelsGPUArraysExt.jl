module WeightedDataRobustModelsGPUArraysExt

import WeightedData: loglikelihood, WeightedValue
import RobustModels: LossFunction, rho
import ZippedArrays: ZippedArray
import GPUArrays: AnyGPUArray

WeightedArrayGPU{T, N}  = ZippedArray{T,N,2,I,Tuple{A, A}} where {T,N,I,A <: AnyGPUArray}

function loglikelihood(loss::LossFunction, data::WeightedArrayGPU{T1, N}, model::AbstractArray{T2, N}) where {T1, T2, N}
           return mapreduce((value, precision, model) -> rho(loss, sqrt(precision) * (model - value)), +, data.args..., model; init = zero(T2))
end

end