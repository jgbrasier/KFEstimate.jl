using Revise, KFEstimate
using LinearAlgebra, Plots
pathof(KFEstimate)

## Normal Kalman Filter

# dynamic model x = [p, v]
dt = 0.01
A = [1.0 dt; 0.0 1.0]
B = zeros(2, 2)
W = 0.01*Matrix{Float64}(I, 2, 2)
dyn = LinearDynamicModel(A, B, W)

# observation model, assume we can noisily measure position
H = [1.0 0]
R_gt = 100.0*Matrix{Float64}(I, 1, 1)
obs = LinearObservationModel(H)

kf = KalmanFilter(dyn, obs)

# run simulation
time_step = 0.0:dt:1000
x0 = [0.0, 1.0]
action_sequence = [[0.0; 0.0] for t in time_step]
sim_states, sim_measurements = run_simulation(kf, R_gt, x0, action_sequence)

# run kalman filter
P0 = Matrix{Float64}(I, 2, 2)
s0 = State(x0, P0) # initial state belief
filtered_states = run_filter(kf, R_gt, s0, action_sequence, sim_measurements)

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
## Noise covariance estimation

R_range = 90:0.1:100
loss = compute_loss(kf, R_range, s0, action_sequence, sim_measurements)

plot(R_range, loss)
