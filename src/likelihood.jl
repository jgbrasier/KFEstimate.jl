""" Kalman Filter and Extended Kalman Filter Likelihoods """


function kf_likelihood(θ, param_kf::ParamKalmanFilter, state_beliefs::AbstractArray,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # drop initial s0 belief
    @assert length(action_history) == length(measurement_history)
    N = length(measurement_history)
    # initialize log likelihood
    # l = o.R[1]
    l=0.0
    for (k, (s, y, u)) in enumerate(zip(state_beliefs, measurement_history, action_history))
        x_hat = param_kf.A(θ)*s.x + param_kf.B(θ)*u # predicted state prior
        P_hat = param_kf.A(θ)*s.P*param_kf.A(θ)' + param_kf.Q(θ) # a priori state covariance
        v = y - param_kf.H(θ)*x_hat # measurement pre fit residual
        S = param_kf.H(θ)*P_hat*param_kf.H(θ)' + param_kf.R(θ) # pre fit residual covariance
        l += 1/2*(v'*inv(S)*v + log(det(S)))
    end
    return l
end


function ekf_likelihood(θ, param_ekf::ExtendedParamKalmanFilter, state_beliefs::AbstractArray,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # drop initial s0 belief
    @assert length(action_history) == length(measurement_history)
    N = length(measurement_history)
    # initialize log likelihood
    # l = o.R[1]
    l=0.0
    for (k, (s, y, u)) in enumerate(zip(state_beliefs, measurement_history, action_history))
        x_hat = param_ekf.f(θ, s.x, u) # predicted state prior
        F = ForwardDiff.jacobian(μ -> param_ekf.f(θ, μ, u), s.x)
        P_hat = F*s.P*F' + param_ekf.Q(θ) # a priori state covariance
        v = y - param_ekf.h(θ, x_hat) # measurement pre fit residual
        H = ForwardDiff.jacobian(μ -> param_ekf.h(θ, μ), s.x)
        S = H*P_hat*H' + param_ekf.R(θ) # pre fit residual covariance
        l += 1/2*(v'*inv(S)*v + log(det(S)))
    end
    return l
end
