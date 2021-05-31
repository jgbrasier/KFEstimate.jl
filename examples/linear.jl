using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux.Optimise
pathof(KFEstimate)
## Normal Kalman Filter

# dynamic model x = [p, v]
dt = 0.001
A = [1.0 dt; 0.0 1.0]
B = zeros(2, 2)
Q = 0.01*Matrix{Float64}(I, 2, 2)

# observation model, assume we can noisily measure position
H = [1.0 0]
R = 10.0*Matrix{Float64}(I, 1, 1)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0.0:dt:10
x0 = [0.0; 10.0]
action_sequence = [[0.0; 0.0] for t in time_step]
sim_states, sim_measurements = run_simulation(kf, x0, action_sequence)
# run kalman filter
P0 = Matrix{Float64}(I, 2, 2)
s0 = State(x0, P0) # initial state belief
filtered_states = run_filter(kf, s0, action_sequence, sim_measurements)

# unpack sim and filtered states
μ, Σ = unpack(filtered_states)
p = [x[1] for x in sim_states]
v = [x[2] for x in sim_states]
# plot
plot(time_step, p[2:end], label = "simulated p", legend=:bottomleft)
plot!(time_step, v[2:end], label = "simulated v", legend=:bottomleft)
plot!(time_step, μ[2:end, 1], label = "measured p", legend=:bottomleft)
plot!(time_step, μ[2:end, 2], label = "estimated v", legend=:bottomleft)
xlabel!("time step (t)")
## Noise covariance loss

R_range = 5:0.1:15
loss = compute_noise_loss(kf, R_range, s0, action_sequence, sim_measurements)
plot(R_range, loss)

## Noise covariance estimation

opt = Optimise.ADAM(0.5)
n_epochs = 20
R_est = 7.0*Matrix{Float64}(I, 1, 1)
estimated_kf = KalmanFilter(A, B, Q, H, R_est)
history = run_noise_estimation(estimated_kf, opt, n_epochs, s0, action_sequence, sim_measurements)
p1 = plot(1:n_epochs, history[:, 1], label="loss")
p2 = plot(1:n_epochs, history[:, 2], label="R")
plot(p1, p2)

R = history[argmin(history[:, 1]), 2]

## Process matrix diagonal coefficients loss

function compute_linear_process_loss(filter::KalmanFilter, param_range::StepRangeLen, s0::State,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # compute loss (log-likelihood) for a given range of process matrix diagonal parameters
    loss = []
    @assert length(action_history) == length(measurement_history)
    for p in param_range
        A = [p 0.001; 0.0 p]
        filter.A = A
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(kf, filtered_states, action_sequence, measurement_history)
        push!(loss, l)
    end
    return loss
end

param_range = 0.5:0.01:1.5
loss = compute_linear_process_loss(kf, param_range, s0, action_sequence, sim_measurements)
plot(param_range, loss)

## Process matrix parameter estimation
opt = Optimise.ADAM(0.5)
n_epochs = 20
A_est = [2.0 0.01; 0.0001 2.0]
estimated_kf = KalmanFilter(A_est, B, Q, H, R)
history = run_linear_estimation(estimated_kf, opt, n_epochs, s0, action_sequence, sim_measurements)

##
using Zygote
estimated_kf = KalmanFilter(A_est, B, Q, H, R)
gs = gradient(f -> likelihood(f, filtered_states, action_sequence, sim_measurements), estimated_kf)[1][]

function heatgif(A::AbstractArray{<:Number,2}; kwargs...)
    p = heatmap(zeros(size(A); kwargs...))
    anim = @animate for i=1:length(A)
        heatmap!(p[1], A[i])
    end
    return anim
end
