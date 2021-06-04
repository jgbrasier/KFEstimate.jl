using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux.Optimise
pathof(KFEstimate)
## Normal Kalman Filter

# dynamic model x = [p, v]
dt = 0.001
A = [1.0 dt 1/2*dt^2; 0.0 1.0 dt; 0.0 0.0 1.0]
B = Matrix{Float64}(I, 3, 3)
Q = 0.1*Matrix{Float64}(I, 3, 3)

# observation model, assume we can noisily measure position
H = [0.0 0.0 1.0]
R = 10.0*Matrix{Float64}(I, 1, 1)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0.0:1000
x0 = [0.0; 0.0; 0.0]
action_sequence = [[0.0; 0.0; exp(-0.01*t)] for t in time_step]
sim_states, sim_measurements = run_simulation(kf, x0, action_sequence)
# run kalman filter
P0 = Matrix{Float64}(I, 3, 3)
s0 = State(x0, P0) # initial state belief
filtered_states = run_filter(kf, s0, action_sequence, sim_measurements)

# unpack sim and filtered states
μ, Σ = unpack(filtered_states)
p = [x[1] for x in sim_states]
v = [x[2] for x in sim_states]
a = [x[3] for x in sim_states]
# plot
plot(time_step, [p[2:end] v[2:end] a[2:end]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomleft)
plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["estimated p" "estimated v" "measured a"], legend=:bottomleft)
xlabel!("time step (t)")
## Plot noise covariance loss

R_range = 5:0.1:15
loss = compute_noise_loss(kf, R_range, s0, action_sequence, sim_measurements)
plot(R_range, loss)

## Noise covariance estimation
# after all states k=1:T, perform gradient descent on noise log-likelihood
opt = Optimise.ADAM(0.5)
n_epochs = 20
R_est = 7.0*Matrix{Float64}(I, 1, 1) # estimation of noise
estimated_kf = KalmanFilter(A, B, Q, H, R_est)
history = run_noise_estimation(estimated_kf, opt, n_epochs, s0, action_sequence, sim_measurements)
plot(1:n_epochs, history["loss"], label="loss")

R = history["R"][argmin(history["loss"])]

## Process matrix parameter estimation
# for each state k, perform gradient descent on process log-likelihood
opt = Optimise.ADAM(0.001)
n_epochs = 10
A_est = randn(3, 3) # estimation of process matrix
estimated_kf = KalmanFilter(A_est, B, Q, H, R)
history = run_linear_estimation(estimated_kf, opt, n_epochs, s0, action_sequence, sim_measurements)
# grads = gradient(f -> loss(f, s0, action_sequence[1], sim_measurements[1]), estimated_kf)[1][]

history = unpack_history(history)

plot(time_step, [history["A"][1, 1, :] history["A"][2, 2, :] history["A"][3, 3, :]], label=["A_11" "A_22" "A_33"])
plot!(time_step, [history["A"][1, 2, :] history["A"][2, 3, :]], label=["A_12" "A_23"])
plot!(time_step, history["A"][1, 3, :], label="A_13")
# plot!(time_step, [A[1, 1]*ones(length(time_step)) A[1, 2]*ones(length(time_step)) A[2, 1]*ones(length(time_step)) A[2, 2]*ones(length(time_step))], ls=[:dash])

## Compare results

est_filtered_states = run_filter(estimated_kf, s0, action_sequence, sim_measurements)
μ_est, Σ_est = unpack(est_filtered_states)
# plot
plot(time_step, [p[2:end] v[2:end] a[2:end]], label = ["simulated p" "simulated v" "simulated a"])
plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["estimated p" "estimated v" "measured a"])
plot!(time_step, μ_est[2:end, 1], label = "estimated p with A_est", legend=:bottomright)
plot!(time_step, μ_est[2:end, 2], label = "estimated v with A_est")
plot!(time_step, μ_est[2:end, 3], label = "measured a with A_est")
xlabel!("time step (t)")
