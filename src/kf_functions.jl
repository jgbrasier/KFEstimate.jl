""" Dynamic """

function dynamic(kf::KalmanFilter, x::AbstractVector, u::AbstractVector)
    return kf.A*x + kf.B*u + cholesky(kf.Q).L*randn(size(kf.Q, 1))
end

function prediction(kf::KalmanFilter, s::State, u::AbstractVector)
    xp = kf.A*s.x + kf.B*u # predicted state prior
    Pp = kf.A*s.P*kf.A' + kf.Q # a priori state covariance
    return State(xp, Pp)
end

""" Observation """

function observation(kf::KalmanFilter, x::AbstractVector)
    return kf.H*x + cholesky(kf.R).L*randn(size(kf.R, 1))
end

function correction(kf::KalmanFilter, s::State, y::AbstractVector)
    v = y - kf.H*s.x # measurement pre fit residual
    S = kf.H*s.P*kf.H' + kf.R # pre fit residual covariance
    K = s.P*kf.H'*inv(S) # Kalman gain
    x_hat = s.x + K*v # a posteriori state estiamate
    P = (I - K*kf.H)*s.P # a posteriori covariance estiamate
    return State(x_hat, P)
end

function pre_fit(kf::KalmanFilter, s::State, u::AbstractVector, y::AbstractVector)
    x = kf.A*s.x + kf.B*u # predicted state prior
    P = kf.A*s.P*kf.A' + kf.Q
    v = y - kf.H*x # measurement pre fit residual
    S = kf.H*P*kf.H' + kf.R # pre fit residual covariance
    return v'*inv(S)*v + log(det(2*π*S)) # log likelihood for a state k
end
