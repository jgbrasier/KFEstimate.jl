""" Linear Kalman Filter Structure """

abstract type AbstractFilter end

mutable struct KalmanFilter{a<:AbstractMatrix, b<:AbstractMatrix, q<:Symmetric, h<:AbstractMatrix, r<:Symmetric} <: AbstractFilter
    A::a # process matrix
    B::b # control matrix
    Q::q # process zero mean noise covariance
    H::h # measurement matrix
    R::r # measurement zero mean noise covariance
end

function KalmanFilter(A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix,
    H::AbstractMatrix, R::AbstractMatrix)
    return KalmanFilter(A, B, Symmetric(Q), H, Symmetric(R))
end

function KalmanFilter(A::AbstractMatrix, Q::AbstractMatrix,
    H::AbstractMatrix, R::AbstractMatrix)
    return KalmanFilter(A, zeros(size(A)), Symmetric(Q), H, Symmetric(R))
end

""" Extended Kalman Filter Structure """

mutable struct ExtendedKalmanFilter{a<:Function, q<:Symmetric, b<:Function, r<:Symmetric} <: AbstractFilter
    f::a # non linear process function f(x, u)
    Q::q # process zero mean noise covariance
    h::b # non linear observation function h(x)
    R::r# measurement zero mean noise covariance
end

function ExtendedKalmanFilter(f::Function, Q::AbstractMatrix, h::Function, R::AbstractMatrix)
    return ExtendedKalmanFilter(f, Symmetric(Q), h, Symmetric(R))
end

""" State: mean and cov """
struct State{a<:Number, b<:Number}
    x::AbstractVector{a}
    P::Symmetric{b}
end

function State(x::AbstractVector, P::AbstractMatrix)
    return State(x, Symmetric(P))
end
