
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
        sp = prediction(filter, states[end], u)
        sn = correction(filter, sp, y)
        push!(states, sn)
    end
    return states
end

function run_noise_estimation(filter::AbstractFilter, opt, n_epochs::Integer, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run a SGD type optimisation on log-likelihood of noise covariance, with initial estimate R0.
    """
    history = zeros(n_epochs, 2) # [loss, R]*n_epochs
    epochs = 1:n_epochs
    @assert length(action_history) == length(measurement_history)
    for e in ProgressBar(epochs)
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(filter, filtered_states, action_history, measurement_history)
        gs = gradient(f -> likelihood(f, filtered_states, action_history, measurement_history), filter)[1][]
        update!(opt, filter.R, gs[:R])
        history[e, 1] = l
        history[e, 2] = filter.R[1]
    end
    return history
end

function update!(hist::AbstractArray{A}, filter::KalmanFilter, l::Float64) where {A<:AbstractArray}
    push!(hist[1], l)
    push!(hist[2], filter.A)
    push!(hist[3], filter.B)
    push!(hist[4], filter.Q)
    push!(hist[5], filter.H)
    push!(hist[6], filter.R)
end

function run_linear_estimation(filter::KalmanFilter, opt, n_epochs::Integer, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run a SGD type optimisation on log-likelihood of noise covariance, with initial estimate R0.
    """
    history = [[] for i in 1:length(fieldnames(typeof(filter)))+1] # [loss, A, B, Q, H, R]
    epochs = 1:n_epochs
    @assert length(action_history) == length(measurement_history)
    for e in ProgressBar(epochs)
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(filter, filtered_states, action_history, measurement_history)
        gs = gradient(f -> likelihood(f, filtered_states, action_history, measurement_history), filter)[1][]
        update!(opt, filter.A, gs[:A])
        update!(history, filter, l)
    end
    return history
end
