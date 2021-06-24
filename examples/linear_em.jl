using KFEstimate
using LinearAlgebra, Plots, Zygote, Statistics, Revise
using Flux, Flux.Optimise
using ProgressBars
pathof(KFEstimate)

##
dt = 0.001
A = [1.0 dt 1/2*dt^2; 0.0 1.0 dt; 0.0 0.0 1.0]
B = [0.0; 0.0; 1.0]
B = reshape(B, length(B), 1)
Q = 0.01*Matrix{Float64}(I, 3, 3)

# observation model, assume we can noisily measure position
H = randn(3, 3)
# H = [1 0 0; 0 1 0; 0 0 1]
R = 10.0*Matrix{Float64}(I, 3, 3)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0:1000
x0 = [0.0; 0.0; 0.0]
action_sequence = [[exp(-0.01*(t-1))] for t in time_step]
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
# plot
plot(time_step, [x[2:end, 1] x[2:end, 2] x[2:end, 3]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomright)
plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtred p" "filtered v" "filtered a"], legend=:bottomright)
xlabel!("time step (t)")

##

# parametrized matrix estimates
Ahat(θ) = [θ[1] dt 1/2*dt^2; 0.0 θ[1] dt; 0.0 0.0 θ[1]]
Bhat(θ) = B
Qhat(θ) = Q
Hhat(θ) = H
Rhat(θ) = R
# define a parametrized kalman filter
param_kf = ParamKalmanFilter(Ahat, Bhat, Qhat, Hhat, Rhat)


θ0 = [1.1]
pkf = ParamKalmanFilter(Ahat, Bhat, Qhat, Hhat, Rhat)

loss = []
θ_range = 0.9:0.001:1.1
for i in θ_range
    θ_i = [i]
    states = run_param_filter(θ_i, pkf, s0, action_sequence, sim_measurements)
    l = kf_likelihood(θ_i, param_kf, states, action_sequence, sim_measurements)
    push!(loss, l)
end

plot(θ_range, loss)

##

# parametrized matrix estimates
Ahat(θ) = [θ[1] θ[2] θ[3]; 0.0 θ[1] θ[2]; 0.0 0.0 θ[1]]
Bhat(θ) = B
Qhat(θ) = Q
Hhat(θ) = H
Rhat(θ) = R
# define a parametrized kalman filter
param_kf = ParamKalmanFilter(Ahat, Bhat, Qhat, Hhat, Rhat)

θ0 = [1.1, 0.002, 4.0e-7]
opt = ADAM(0.001)
epochs = 500
newθ, loss = run_gradient(θ0, pkf, s0, action_sequence, sim_measurements, opt, epochs)

grad_states = run_param_filter(newθ, pkf, s0, action_sequence, sim_measurements)
μgrad, Σgrad = unpack(grad_states)

l = @layout [a{0.7h};grid(1, 3)]
p1 = plot(time_step, [x[2:end, 1] x[2:end, 2] x[2:end, 3]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomright, xlabel="time step (t)")
p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtered p" "filtered v" "filtered a"], legend=:bottomright, xlabel="time step (t)")
p1 = plot!(time_step, [μgrad[2:end, 1] μgrad[2:end, 2] μgrad[2:end, 3]], label = ["learned p" "learned v" "learned a"], legend=:bottomright, xlabel="time step (t)")
p2 = plot(1:epochs, loss, title="loss", xlabel="number of epochs")
p3 = plot(time_step, (x[2:end, :]-μ[2:end, :]).^2, title="KF vs. sim error", xlabel="time step (t)")
p4 = plot(time_step, (x[2:end, :]-μgrad[2:end, :]).^2, title="grad vs. sim error", xlabel="time step (t)")
plot(p1, p2, p3, p4, layout=l, titlefont = font(12), size=(1000, 700))
