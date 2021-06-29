""" Dynamic """

function dynamic(ekf::ExtendedKalmanFilter, x::AbstractVector, u::AbstractVector)
    # Non linear function F has to have x and u as inputs
    return ekf.f(x, u) + cholesky(ekf.Q).L*randn(size(ekf.Q, 1))
end

function prediction(ekf::ExtendedKalmanFilter, s::State, u::AbstractVector)
    xp = ekf.f(s.x, u) # predicted state prior
    F = ForwardDiff.jacobian(μ -> ekf.f(μ, u), s.x)
    Pp = F*s.P*F' + ekf.Q # a priori state covariance
    return State(xp, Pp)
end

function param_dynamic(θ, parma_ekf::ExtendedParamKalmanFilter, x::AbstractVector, u::AbstractVector)
    # Non linear function F has to have x and u as inputs
    return param_ekf.f(θ, x, u) + cholesky(param_ekf.Q(θ)).L*randn(size(parma_ekf.Q(θ), 1))
end

function param_prediction(θ, param_ekf::ExtendedParamKalmanFilter, s::State, u::AbstractVector)
    xp = param_ekf.f(θ, s.x, u) # predicted state prior
    F = ForwardDiff.jacobian(μ -> param_ekf.f(θ, μ, u), s.x)
    Pp = F*s.P*F' + param_ekf.Q(θ) # a priori state covariance
    return State(xp, Pp)
end

""" Observation """

function observation(ekf::ExtendedKalmanFilter, x::AbstractVector)
    return ekf.h(x) + cholesky(ekf.R).L*randn(size(ekf.R, 1))
end

function correction(ekf::ExtendedKalmanFilter, s::State, y::AbstractVector)
    v = y - ekf.h(s.x) # measurement pre fit residual
    H = ForwardDiff.jacobian(μ -> ekf.h(μ), s.x)
    S = H*s.P*H' + ekf.R # pre fit residual covariance
    K = s.P*H'*inv(S) # Kalman gain
    x_hat = s.x + K*v # a posteriori state estiamate
    P_hat = (I - K*H)*s.P # a posteriori covariance estiamate
    return State(x_hat, P_hat)
end

function param_observation(θ, param_ekf::ExtendedParamKalmanFilter, x::AbstractVector)
    return param_ekf.h(θ, x) + cholesky(param_ekf.R(θ)).L*randn(size(param_ekf.R(θ), 1))
end

function param_correction(θ, param_ekf::ExtendedParamKalmanFilter, s::State, y::AbstractVector)
    v = y - param_ekf.h(θ, s.x) # measurement pre fit residual
    H = ForwardDiff.jacobian(μ -> param_ekf.h(θ, μ), s.x)
    S = H*s.P*H' + param_ekf.R(θ) # pre fit residual covariance
    K = s.P*H'*inv(S) # Kalman gain
    x_hat = s.x + K*v # a posteriori state estiamate
    P_hat = (I - K*H)*s.P # a posteriori covariance estiamate
    return State(x_hat, P_hat)
end

"""Loss """

function state_mse(filter::ExtendedKalmanFilter, s::State, u::AbstractVector, y::AbstractVector)
    ϵx = norm(s.x - filter.f(s.x))
    ϵy = norm(y - filter.h(s.x))
    return ϵx + ϵy
end
