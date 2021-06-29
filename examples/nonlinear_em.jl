using KFEstimate
using LinearAlgebra, Plots, Revise, ProgressBars
using Flux, Flux.Optimise
using Flux: Params, gradient
pathof(KFEstimate)
global dt = 0.001
g = 9.81

# non linear transition functions must always take x and u as variables regardless if they are used
function f(x, u)
    return [x[1] + x[2]*dt; x[2] - g*dt*sin(x[1])]
end

# non linear observation function only takes x as an input variable
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

function fhat(θ, x, u)
    return [x[1] + x[2]*θ[1]; x[2] - θ[2]*sin(x[1])]
end

function hhat(θ, x)
    return [sin(x[1])]
end

Rhat(θ) = R
Qhat(θ) = Q


param_ekf = ExtendedParamKalmanFilter(fhat, Qhat, hhat, Rhat)

opt = ADAM(0.005)
epochs = 100

θ0 = [0.001, 0.002]
newθ, loss = run_ekf_gradient(θ0, param_ekf, s0, action_sequence, sim_measurements, opt, epochs)


grad_states = run_param_filter(newθ, param_ekf, s0, action_sequence, sim_measurements)
μgrad, Σgrad = unpack(grad_states)

l = @layout [a{0.7h};grid(1, 3)]
p1 = plot(time_step, [θ[2:end, 1] θ[2:end, 2]], label = ["simulated θ" "simulated dθ"], legend=:bottomright, xlabel="time step (t)")
p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2] ], label = ["filtered θ" "filtered dθ"], legend=:bottomright, xlabel="time step (t)")
p1 = plot!(time_step, [μgrad[2:end, 1] μgrad[2:end, 2]], label = ["learned θ" "learned dθ"], legend=:bottomright, xlabel="time step (t)")
p2 = plot(1:epochs, loss, title="loss", xlabel="number of epochs")
p3 = plot(time_step, (θ[2:end, :]-μ[2:end, :]).^2, title="KF vs. sim error", xlabel="time step (t)")
p4 = plot(time_step, (θ[2:end, :]-μgrad[2:end, :]).^2, title="grad vs. sim error", xlabel="time step (t)")
plot(p1, p2, p3, p4, layout=l, titlefont = font(12), size=(1000, 700))
