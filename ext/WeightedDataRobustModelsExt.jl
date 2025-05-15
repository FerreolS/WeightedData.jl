module WeightedDataRobustModelsExt

import WeightedData: likelihood, WeightedPoint, get_data, get_precision

import RobustModels: LossFunction,
    BoundedLossFunction,
    ConvexLossFunction,
    CompositeLossFunction,
    L2Loss,
    L1Loss,
    HuberLoss,
    L1L2Loss,
    FairLoss,
    LogcoshLoss,
    ArctanLoss,
    CatoniWideLoss,
    CatoniNarrowLoss,
    CauchyLoss,
    GemanLoss,
    WelschLoss,
    TukeyLoss,
    YohaiZamarLoss,
    HardThresholdLoss,
    HampelLoss,
    loss, rho



function likelihood(loss::LossFunction, data::AbstractArray{WeightedPoint{T},N}, model::AbstractArray{T,N}) where {T,N}
    size(data) == size(model) || error("likelihood : size(A) != size(model)")
    r = @. sqrt($get_precision(data)) * (model - $get_data(data))
    l = Base.Fix1(rho, loss)


    return mapreduce(l, +, r)
end

function likelihood(loss::LossFunction, data::WeightedPoint, model::Number)
    r = sqrt(get_precision(data)) * (model - get_data(data))
    return rho(loss, r)
end
end