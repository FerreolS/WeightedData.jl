
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
Base.:+(A::Number, B::WeightedPoint)  = B + A
Base.:-(A::WeightedPoint, B::WeightedPoint)  = WeightedPoint(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
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
