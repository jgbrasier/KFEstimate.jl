""" Dynamic """

function dynamic(m::NonLinearDynamicModel, x::AbstractVector, u::AbstractVector)
    # Non linear function F has to have x and u as inputs
    return m.F(x, u) + cholesky(m.W).L*randn(size(m.W, 1))
end

function prediction(m::NonLinearDynamicModel, s::State, u::AbstractVector)
    xp = m.F(s.x, u) # predicted state prior
    F = ForwardDiff.jacobian(μ -> m.F(μ, u), s.x)
    Pp = F*s.P*F' + m.W # a priori state covariance
    return State(xp, Pp)
end

""" Observation """

function observation(m::NonLinearObservationModel, R::AbstractMatrix, x::AbstractVector)
    return m.H(x) + cholesky(R).L*randn(size(R, 1))
end

function correction(m::NonLinearObservationModel, R::AbstractMatrix, s::State, y::AbstractVector)
    v = y - m.H(s.x) # measurement pre fit residual
    H = ForwardDiff.jacobian(μ -> m.H(μ), s.x)
    S = H*s.P*H' + R # pre fit residual covariance
    K = s.P*H'*inv(S) # Kalman gain
    x_hat = s.x + K*v # a posteriori state estiamate
    P = (I - K*H)*s.P # a posteriori covariance estiamate
    return State(x_hat, P)
end

function pre_fit(m::NonLinearObservationModel, R::AbstractMatrix, s::State, y::AbstractVector)
    v = y - m.H(s.x) # measurement pre fit residual
    H = ForwardDiff.jacobian(μ -> m.H(μ), s.x)
    S = H*s.P*H' + R # pre fit residual covariance
    return v'*inv(S)*v + log(det(2*π*S)) # log likelihood for a state k
end
