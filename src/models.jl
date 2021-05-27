""" State: mean and cov """
struct State{a<:Number, b<:Number}
    x::AbstractVector{a}
    P::Symmetric{b}
end

function State(x::AbstractVector, P::AbstractMatrix)
    return State(x, Symmetric(P))
end

""" dynamic models """
abstract type DynamicModel end

struct LinearDynamicModel{a<:AbstractMatrix, b<: AbstractMatrix, q<:Symmetric} <:DynamicModel
    A::a
    B::b
    W::q
end

function LinearDynamicModel(A::AbstractMatrix, B::AbstractMatrix, W::AbstractMatrix)
    return LinearDynamicModel(A, B, Symmetric(W))
end

""" observation models"""
abstract type ObservationModel end


struct LinearObservationModel{c<:AbstractMatrix} <:ObservationModel
    # noise covariance R is not defined because it is a parameter to estimate
    H::c
end
