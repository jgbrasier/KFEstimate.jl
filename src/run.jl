
function run_simulation(filter::AbstractFilter, x0::AbstractVector, action_seq::AbstractArray)
    """ Simulate N=length(action_seq) points correspding to a
    Dynamic/Observation Model encapsulated in a KF struct.
    Start from an initial state belief s0"""
    meas = Vector{typeof(x0)}()
    states = [x0]
    for u in action_seq
        xp = dynamic(filter, states[end], u)
        y = observation(filter, xp)
        push!(meas, y)
        push!(states, xp)
    end
    return states, meas
end


function run_filter(filter::AbstractFilter, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run Filter on a measurement_history points, for a given action_history
    Start from an initial state belief s0
    """
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    for (u, y) in zip(action_history, measurement_history)
        # filter.A += rand(Normal(0, 0.005), (3, 3))
        sp = prediction(filter, states[end], u)
        sn = correction(filter, sp, y)
        push!(states, sn)
    end
    return states
end

function run_param_filter(θ, param_filter::AbstractParamFilter, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run parametrized Klaman Filter on a measurement_history points, for a given action_history
    Start from an initial state belief s0
    """
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    for (u, y) in zip(action_history, measurement_history)
        sp = param_prediction(θ, param_filter, states[end], u)
        sn = param_correction(θ, param_filter, sp, y)
        push!(states, sn)
    end
    return states
end

function run_kf_gradient(θ, param_kf::ParamKalmanFilter, s0::State, action_history::AbstractArray, measurement_history::AbstractArray,
    opt, epochs)
    """
    Run gradient descent on unknown parameters of a linear on non linear state space model (parametrized abstract filter).
    returns the found paramters after n epochs, and the associated loss function
    """
    @assert length(action_history)==length(measurement_history)
    # Compute initial state esimates
    loss = []
    for i in ProgressBar(1:epochs)
        states = run_param_filter(θ, param_kf, s0, action_history, measurement_history)
        ∇, = gradient(theta -> kf_likelihood(theta, param_kf, states, action_history, measurement_history), θ)
        update!(opt, θ, ∇)
        l = kf_likelihood(θ, param_kf, states, action_history, measurement_history)
        push!(loss, l)
    end
    return θ, loss
end

function run_ekf_gradient(θ, param_ekf::ExtendedParamKalmanFilter, s0::State, action_history::AbstractArray, measurement_history::AbstractArray,
    opt, epochs)
    """
    Run gradient descent on unknown parameters of a linear on non linear state space model (parametrized abstract filter).
    returns the found paramters after n epochs, and the associated loss function
    """
    @assert length(action_history)==length(measurement_history)
    # Compute initial state esimates
    loss = []
    for i in ProgressBar(1:epochs)
        states = run_param_filter(θ, param_ekf, s0, action_history, measurement_history)
        ∇, = gradient(theta -> ekf_likelihood(theta, param_ekf, states, action_history, measurement_history), θ)
        update!(opt, θ, ∇)
        l = ekf_likelihood(θ, param_ekf, states, action_history, measurement_history)
        push!(loss, l)
    end
    return θ, loss
end

function run_online_kf_gradient(θ, param_kf::ParamKalmanFilter, s0::State, action_history::AbstractArray, measurement_history::AbstractArray,
    opt, epochs)
    """
    Run online gradient descent on unknown parameters of a linear state space model (parametrized abstract filter).
    returns the found paramters after n epochs.
    """
    @assert length(action_history)==length(measurement_history)
    # Compute initial state esimates
    # θgt = [1.0 dt 1/2*dt^2]
    # err = [θgt.-θ]
    states = [s0]
    for (u, y) in zip(action_history, measurement_history)
        l = 0
        s = states[end]
        for i in 1:epochs
            ∇, = gradient(θ -> online_kf_likelihood(θ, param_kf, s, u, y), θ)
            update!(opt, θ, ∇)
        end
        # push!(err, (θgt.-θ).^2)
        sp = param_prediction(θ, param_kf, s, u)
        sn = param_correction(θ, param_kf, sp, y)
        push!(states, sn)
    end
    return states #, err
end
