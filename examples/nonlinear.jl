using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux.Optimise
pathof(KFEstimate)

## pendulum simulation

dt = 0.0001
g = 9.81

function f(x, u)
    dx = x + dt*u
    dx[1] = x[1] + x[2]*dt
    dx[2] = x[2] - g*dt*sin(x[1])
    return dx
end

function h(x)
    return [sin(x[1])]
end

Q = 0.1*[dt^3/3 dt^2/2; dt^2/2 dt]
R_gt = 10.0*Matrix{Float64}(I, 1, 1)

ekf = ExtendedKalmanFilter(f, Q, h, R_gt);
# run simulation
time_step = 0.0:dt:10
x0 = [0.0, 1.0]
action_sequence = [[0.0; 0.0] for t in time_step]
sim_states, sim_measurements = run_simulation(ekf, x0, action_sequence)
# run kalman filter
P0 = Matrix{Float64}(I, 2, 2)
s0 = State(x0, P0) # initial state belief
filtered_states = run_filter(ekf, s0, action_sequence, sim_measurements)

# unpack sim and filtered states
μ, Σ = unpack(filtered_states)
θ = [x[1] for x in sim_states]
dθ= [x[2] for x in sim_states]
# plot
plot(time_step, θ[2:end], label = "simulated θ", legend=:bottomleft)
plot!(time_step, dθ[2:end], label = "simulated dθ", legend=:bottomleft)
plot!(time_step, μ[2:end, 1], label = "measured θ", legend=:bottomleft)
plot!(time_step, μ[2:end, 2], label = "estimated dθ", legend=:bottomleft)
xlabel!("time step (t)")

## Noise covariance loss

R_range = 5:0.1:15
loss = compute_noise_loss(ekf, R_range, s0, action_sequence, sim_measurements)

plot(R_range, loss)

## Noise covariance estimation

opt = Optimise.ADAM(0.5)
n_epochs = 20
R0 = 8.0*Matrix{Float64}(I, 1, 1)
history = run_noise_estimation(ekf, opt, n_epochs, s0, action_sequence, sim_measurements)
p1 = plot(1:n_epochs, history[:, 1], label="loss")
p2 = plot(1:n_epochs, history[:, 2], label="R")
plot(p1, p2)
##
