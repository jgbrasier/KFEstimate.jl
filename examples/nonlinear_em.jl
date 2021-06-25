using KFEstimate
using LinearAlgebra, Plots, Revise, ProgressBars
using Flux, Flux.Optimise
using Flux: Params, gradient
pathof(KFEstimate)

## pendulum simulation

global dt = 0.001
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

Q = 0.01*[dt^3/3 dt^2/2; dt^2/2 dt]
R = 1.0*Matrix{Float64}(I, 1, 1)

ekf = ExtendedKalmanFilter(f, Q, h, R);
# run simulation
time_step = 0.0:5000
x0 = [pi/2, 0.0]
P0 = Matrix{Float64}(I, 2, 2)
action_sequence = [[0.0; 0.0] for t in time_step]
sim_states, sim_measurements = run_simulation(ekf, x0, action_sequence)
# run kalman filter
s0 = State(x0, P0) # initial state belief
filtered_states = run_filter(ekf, s0, action_sequence, sim_measurements)

# unpack sim and filtered states
μ, Σ = unpack(filtered_states)
θ = hcat([x[1] for x in sim_states], [x[2] for x in sim_states])
# plot
plot(time_step, θ[2:end, 1], label = "simulated θ", legend=:bottomleft)
plot!(time_step, θ[2:end, 2], label = "simulated dθ", legend=:bottomleft)
plot!(time_step, μ[2:end, 1], label = "filterd θ", legend=:bottomleft)
plot!(time_step, μ[2:end, 2], label = "filterd dθ", legend=:bottomleft)
xlabel!("time step (t)")

##

hidden_dim = 16
fθ = Chain(Dense(length(x0), hidden_dim, sigmoid), Dense(hidden_dim, length(x0)))
