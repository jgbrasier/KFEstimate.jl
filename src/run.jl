
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
    Run a SGD type optimisation on log-likelihood of linear model, with initial state estimate s0.
    """
    history = init_history(filter) # [loss, A, B, Q, H, R]
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    s_learned = [s0]
    for (u, y) in ProgressBar(zip(action_history, measurement_history))
        sp = prediction(filter, states[end], u)
        sn = correction(filter, sp, y)
        x_learn = [0.0 0.0]
        for i in 1:n_epochs
            s = s_learned[end]
            L_grad, A_grad = linear_grad(filter, s, u, y)
            x_learn = s.x - 0.01*L_grad
            filter.A = filter.A - 0.000001*A_grad
            # l = step_loss(filter, s, u, y)
            # grads = gradient(f -> step_loss(f, s, u, y), filter)[1][]
            # update!(opt, filter.A, grads[:A])
        end
        s_learn = State(x_learn, sn.P)
        # s_learn = train_state(filter, opt, n_epochs, s_learned[end], u, y)
        log_kf_history!(history, filter, 0.0)
        push!(states, sn)
        push!(s_learned, s_learn)
    end
    return history
end

""" Run Utils """
# these are local (private functions) not exported by KFEstinate.jl

function init_history(filter::AbstractFilter)
    history = Dict("loss"=>[])
    for field in fieldnames(typeof(filter))
        history[String(field)] = []
    end
    return history
end

function log_kf_history!(hist::Dict, filter::KalmanFilter, l::Float64)
    push!(hist["loss"], [l])
    push!(hist["A"], copy(filter.A))
    push!(hist["B"], copy(filter.B))
    push!(hist["Q"], copy(filter.Q))
    push!(hist["H"], copy(filter.H))
    push!(hist["R"], copy(filter.R))
end

function train_state(filter::KalmanFilter, opt, n_epochs::Integer, s::State, u::AbstractVector, y::AbstractVector)
    x_learn = 0.0
    for i in 1:n_epochs
        L_grad, A_grad = linear_grad(filter, s, u, y)
        x_learn = s.x - 0.01*L_grad
        filter.A = filter.A - 0.000001*A_grad
        # l = step_loss(filter, s, u, y)
        # grads = gradient(f -> step_loss(f, s, u, y), filter)[1][]
        # update!(opt, filter.A, grads[:A])
    end
    return State(x_learn, s.P)
end
