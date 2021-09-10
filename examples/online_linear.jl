## Online update θ
using KFEstimate
using LinearAlgebra, Plots, Revise, ProgressBars
using Flux, Flux.Optimise
using Flux: Params, gradient
pathof(KFEstimate)

dt = 0.001
A = [1.0 dt 1/2*dt^2; 0.0 1.0 dt; 0.0 0.0 1.0]
B = [0.0; 0.0; 1.0]
B = reshape(B, length(B), 1)
Q = 1*Matrix{Float64}(I, 3, 3)

# observation model, assume we can noisily measure position
H = randn(3, 3)
# H = [1 0 0; 0 1 0; 0 0 1]
R = 10.0*Matrix{Float64}(I, 3, 3)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0:1000
x0 = [0.0; 0.0; 0.0]
action_sequence = [[1] for t in time_step]
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
x = hcat(p, v, a)

Ahat(θ) = [θ[1] θ[2] θ[3]; 0 θ[1] θ[2]; 0 0 θ[1]]
# Ahat(θ) = [θ[1, 1] θ[1, 2] θ[1, 3]; 0.0 θ[2, 2] θ[2, 3]; 0.0 0.0 θ[3, 3]]
Bhat(θ) = B
Qhat(θ) = Q
Hhat(θ) = H
Rhat(θ) = R
# define a parametrized kalman filter
param_kf = ParamKalmanFilter(Ahat, Bhat, Qhat, Hhat, Rhat)

θ0 = [1.5 -1 0.2]
θgt = [1.0 dt 1/2*dt^2]
println(θ0)
opt = ADAM(0.01)
epochs = 200

online_states, θ_err = run_online_kf_gradient(θ0, param_kf, s0, action_sequence, sim_measurements, opt, epochs)
μon, Σon = unpack(online_states)

p1 = plot(time_step, [x[2:end, 1] x[2:end, 2] x[2:end, 3]], label = ["simulated p" "simulated v" "simulated a"], xlabel="time step (t)", ylabel="value (arbitrary)")
p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtered p" "filtered v" "filtered a"], xlabel="time step (t)", ylabel="value (arbitrary)")
p1 = plot!(time_step, [μon[2:end, 1] μon[2:end, 2] μon[2:end, 3]], label = ["learned p" "learned v" "learned a"],  xlabel="time step (t)", ylabel="value (arbitrary)", legend=:topleft)
plot(p1, legendfont=font(10), titlefont = font(12), size=(500, 500))


err1 = [x[1] for x in θ_err]
err2 = [x[2] for x in θ_err]
err3 = [x[3] for x in θ_err]
err = hcat(err1, err2, err3)
p2 = plot(time_step, [err[2:end, 1] err[2:end, 2] err[2:end, 3]], label = ["error on θ1" "error on θ2" "error on θ3"], xlabel="time step (t)", ylabel="online error on θ")
plot(p2, legendfont=font(10), size=(500, 500))
