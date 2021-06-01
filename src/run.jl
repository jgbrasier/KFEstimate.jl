
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

function init_history(filter::AbstractFilter, n_epochs::Int)
    history = Dict("loss"=>[])
    for field in fieldnames(typeof(filter))
        history[String(field)] = []
    end
    return history
end

function log_kf_history!(hist::Dict, filter::KalmanFilter, l::Float64, epoch::Int)
    push!(hist["loss"], l)
    push!(hist["A"], filter.A)
    push!(hist["B"], filter.B)
    push!(hist["Q"], filter.Q)
    push!(hist["H"], filter.H)
    push!(hist["R"], filter.R)
end

function run_noise_estimation(filter::AbstractFilter, opt, n_epochs::Integer, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run a SGD type optimisation on log-likelihood of noise covariance, with initial estimate R0.
    """
    history = init_history(filter) # [loss, R]*n_epochs
    epochs = 1:n_epochs
    @assert length(action_history) == length(measurement_history)
    for e in ProgressBar(epochs)
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(filter, filtered_states, action_history, measurement_history)
        gs = gradient(f -> likelihood(f, filtered_states, action_history, measurement_history), filter)[1][]
        update!(opt, filter.R, gs[:R])
        log_kf_history!(history, filter, l)
    end
    return history
end

function run_linear_estimation(filter::KalmanFilter, opt, n_epochs::Integer, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run a SGD type optimisation on log-likelihood of noise covariance, with initial estimate R0.
    """
    history = init_history(filter) # [loss, A, B, Q, H, R]
    epochs = 1:n_epochs
    @assert length(action_history) == length(measurement_history)
    for e in ProgressBar(epochs)
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(filter, filtered_states, action_history, measurement_history)
        gs = gradient(f -> likelihood(f, filtered_states, action_history, measurement_history), filter)[1][]
        update!(opt, filter.A, gs[:A])
        log_history!(history, filter, l)
    end
    return history
end
