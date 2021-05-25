""" Dynamic """

function dynamic(m::LinearDynamicModel, x::AbstractVector, u::AbstractVector)
    return m.A*x + m.B*u + cholesky(m.W).L*randn(rng, Float64, size(m.W, 1))
end

function prediction(m::LinearDynamicModel, s::State)
    xp = m.A*s.x + m.B*u # predicted state prior
    Pp = m.A*s.P*m.A' + m.W # a priori state covariance
    return State(xp, Pp)
end

""" Observation """

function observation(m::LinearObservationModel, R::AbstractMatrix, x::AbstractVector)
    if size(R) > (1, 1)
        return m.H*x + cholesky(R).L*randn(rng, Float64, size(m.V, 1))
    else
        return m.H*x + R*randn(rng, Float64, size(m.V, 1))
    end
end

function correction(m::LinearObservationModel, R::AbstractMatrix, s::State, y::AbstractVector)
    v = y - m.H*s.x # measurement pre fit residual
    S = m.H*s.P*m.H' + R # pre fit residual covariance
    K = s.P*m.H'*inv(S) # Kalman gain
    x_hat = s.x + K*v # a posteriori state estiamate
    P = (I - K*m.H)*s.P # a posteriori covariance estiamate
    return State(x_hat, P)
end

function pre_fit(m::LinearObservationModel, R::AbstractMatrix, s::State, y::AbstractVector)
    v = y - m.H*s.x # measurement pre fit residual
    S = m.H*s.P*m.H' + R # pre fit residual covariance
    return v'*inv(S)*v + log(det(2*π*S)) # log likelihood for a state k
end
