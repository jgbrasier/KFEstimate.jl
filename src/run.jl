
function run_simulation(filter::AbstractFilter, R::AbstractMatrix, x0::AbstractVector, action_seq::AbstractArray)
    """ Simulate N=length(action_seq) points correspding to a
    Dynamic/Observation Model encapsulated in a KF struct.
    Start from an initial state belief s0"""
    meas = Vector{typeof(x0)}()
    states = [x0]
    for u in action_seq
        xp = dynamic(filter.d, states[end], u)
        y = observation(filter.o, R, xp)
        push!(meas, y)
        push!(states, xp)
    end
    return states, meas
end


function run_filter(filter::AbstractFilter, R::AbstractMatrix, s0::State, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run Filter on a measurement_history points, for a given action_history
    Start from an initial state belief s0
    """
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    for (u, y) in zip(action_history, measurement_history)
        sp = prediction(filter.d, states[end], u)
        sn = correction(filter.o, R, sp, y)
        push!(states, sn)
    end
    return states
end

function run_estimation(filter::AbstractFilter, opt, R::AbstractMatrix, s0::State, n_epochs::Integer, action_history::AbstractArray,
    measurement_history::AbstractArray)
    """
    Run a SGD type optimisation on log-likelihood of noise covariance, with initial estimate R0.
    """
    history = zeros(n_epochs, 2) # [loss, R]*n_epochs
    epochs = 1:n_epochs
    @assert length(action_history) == length(measurement_history)
    N = length(measurement_history)
    for e in ProgressBar(epochs)
        filtered_states = run_filter(filter, R, s0, action_history, measurement_history)
        l = likelihood(filter, R, filtered_states, action_history, measurement_history)
        grad, = gradient(r -> likelihood(filter, r, filtered_states, action_history, measurement_history), R)
        update!(opt, R, grad)
        history[e, 1] = l
        history[e, 2] = R[1]
    end
    return history
end
