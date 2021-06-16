""" Dynamic """

function dynamic(kf::KalmanFilter, x::AbstractVector, u::AbstractVector)
    return kf.A*x + kf.B*u + cholesky(kf.Q).L*randn(size(kf.Q, 1))
end

function prediction(kf::KalmanFilter, s::State, u::AbstractVector)
    x_hat = kf.A*s.x + kf.B*u # predicted state prior
    P_hat = kf.A*s.P*kf.A' + kf.Q # a priori state covariance
    return State(x_hat, P_hat)
end

""" Observation """

function observation(kf::KalmanFilter, x::AbstractVector)
    return kf.H*x + cholesky(kf.R).L*randn(size(kf.R, 1))
end

function correction(kf::KalmanFilter, s::State, y::AbstractVector)
    v = y - kf.H*s.x # measurement pre fit residual
    S = kf.H*s.P*kf.H' + kf.R # pre fit residual covariance
    K = s.P*kf.H'*inv(S) # Kalman gain
    x_post = s.x + K*v # a posteriori state estiamate
    P_post = (I - K*kf.H)*s.P # a posteriori covariance estiamate
    return State(x_post, P_post)
end

function pre_fit(kf::KalmanFilter, s::State, u::AbstractVector, y::AbstractVector)
    v = y - kf.H*s.x # measurement pre fit residual
    S = kf.H*s.P*kf.H' + kf.R # pre fit residual covariance
    return v'*inv(S)*v + log(det(2*π*S)) # log likelihood for a state k
end

function likelihood(filter::KalmanFilter, s::State, u::AbstractVector, y::AbstractVector)
    ϵx = s.x - (filter.A*s.x + filter.B*u)
    ϵy = y - filter.H*s.x
    return ϵx'*filter.R*ϵx + ϵy'*s.P*ϵy
end

function mse_loss(filter::KalmanFilter, s::State, u::AbstractVector, y::AbstractVector)
    ϵx = norm(s.x - (filter.A*s.x + filter.B*u))
    ϵy = norm(y - filter.H*s.x)
    return ϵx + ϵy
end
