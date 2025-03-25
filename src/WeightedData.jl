module WeightedData
using ChainRulesCore

export WeightedArray, WeightedPoint

struct WeightedPoint{T<:AbstractFloat} <: AbstractFloat
	val::T
	precision::T
	function WeightedPoint(val::T,precision::T) where T<:AbstractFloat
		precision >= 0 || error("WeightedPoint : precision < 0 ")
		new{T}(val,precision)
	end
end
WeightedPoint(val::T,precision::Number) where {T<:AbstractFloat} = WeightedPoint(val,T(precision))

Base.real(A::WeightedPoint) = WeightedPoint(real.(A.val),real.(A.precision))
Base.imag(A::WeightedPoint) = WeightedPoint(imag.(A.val),imag.(A.precision))

Base.:+(A::WeightedPoint, B::WeightedPoint)  = WeightedPoint(A.val .+ B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:+(A::WeightedPoint, B::Number)  = WeightedPoint(A.val .+ B, A.precision )
Base.:+(A::Number, B::WeightedPoint)  = A+B
Base.:-(A::WeightedPoint, B::WeightedPoint)  = WeightedData(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::WeightedPoint, B::Number)  = WeightedPoint(A.val .- B, A.precision )
Base.:/(A::WeightedPoint, B::Number)  = WeightedPoint(A.val ./ B, B.^2 .* A.precision)
Base.:*(B::Number, A::WeightedPoint)  = WeightedPoint(A.val .* B,  A.precision ./ B.^2 )
Base.:*(A::WeightedPoint, B::Number)  = B * A

function combine(A::WeightedPoint, B::WeightedPoint) 
	precision = A.precision .+B.precision
	val =(A.precision .* A.val .+ B.precision .* B.val)./(precision)
    WeightedPoint(val,precision )
end
combine(B::NTuple{N,WeightedPoint{T}}) where {N,T}  = combine(first(B),last(B, N-1)...)
combine(A::WeightedPoint, B...)   = combine(combine(A,first(B)),last(B, length(B)-1)...)
combine(B::AbstractArray{WeightedPoint{T}})  where T= combine(first(B),last(B, length(B)-1)...)
combine(A::WeightedPoint, B::AbstractArray{WeightedPoint{T}})where T  = combine(combine(A,first(B)),last(B, length(B)-1)...)
combine(A::WeightedPoint) = A


function gausslikelihood((;val,precision)::WeightedPoint{T},model::Number) where {T}
	return (val - model)^2 * precision / 2
end 

function robustlikelihood((;val,precision)::WeightedPoint{T1},model::T2,s::Number) where {T1,T2}
	T = promote_type(T1,T2) 
	γ = T(2.385)
	r = T(s/γ)*sqrt(precision)*( model - val) 
	return log(T(1) + r^2)
end

robustlikelihood(s::Number) = (D::WeightedPoint,model::Number) -> robustlikelihood(D,model,s)


## Array of WeightedPoint

get_val(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x->x.val, x)
get_precision(x::AbstractArray{WeightedPoint{T},N}) where {T,N} = map(x->x.precision, x)

function flagbadpix!(data::AbstractArray{WeightedPoint{T},N},badpix::Union{ Array{Bool, N},BitArray{N}}) where {T,N}
    size(data) == size(badpix) || error("flagbadpix! : size(A) != size(badpix)")
	map!(x->x.val[badpix] .= T(0), data)
	map!(x->x.precision[badpix] .= T(0), data)
end

function likelihood(data::AbstractArray{WeightedPoint{T},N},model::AbstractArray; likelihoodfunc::F=gausslikelihood) where {T,N,F<:Function}
	return likelihood(likelihoodfunc,data,model)
end 
function likelihood(likelihoodfunc::F,data::AbstractArray{WeightedPoint{T},N},model::AbstractArray{T2,N}) where {T,T2,N,F<:Function}
	size(data) == size(model) || error("likelihood : size(A) != size(model)")
	mapreduce( (x,y)-> likelihoodfunc(x,y), +, data, model)
end 


function ChainRulesCore.rrule( ::typeof(WeightedData.likelihood),::typeof(WeightedData.gausslikelihood), data::AbstractArray{WeightedPoint{T},N},model::AbstractArray{T2,N};)  where {T,T2,N}
	r =model .- get_val(data)
	rp = get_precision(data) .* r
	likelihood_pullback(Δy) = (NoTangent(),NoTangent(),NoTangent(), rp .* Δy)
	return  sum(r.*rp) / 2, likelihood_pullback
end


function scaledlikelihood(data::AbstractArray{WeightedPoint{T1},N},model::AbstractArray{T2,N}) where{T1,T2,N}
	size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
	val = get_val(data)
	precision = get_precision(data)

	α = max.(0,sum(model .* precision .* val,dims=2) ./ sum( model .*  precision .* model,dims=2) )
	α[.!isfinite.(α)] .= T2(0)
	res = ( α .* model .- val) 
	return sum(res.^2 .* precision)/2
end

function ChainRulesCore.rrule( ::typeof(scaledlikelihood),data::AbstractArray{WeightedPoint{T1},N},model::AbstractArray{T2,N}) where{T1,T2,N}
	size(data) == size(model) || error("scaledlikelihood : size(A) != size(model)")
	val = get_val(data)
	precision = get_precision(data)

	α = max.(0,sum(model .* precision .* val,dims=2) ./ sum( model .*  precision .* model,dims=2) )

	r =( α .*model .- val)
	rp = r .* precision
    likelihood_pullback(Δy) = (NoTangent(),ZeroTangent(), α .* rp .* Δy)
    return  sum(r.*rp) / 2, likelihood_pullback
end

end
