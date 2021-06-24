using KFEstimate
using LinearAlgebra, Plots, Zygote, Statistics, Revise
using Flux.Optimise
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
Ahat(θ) = [θ dt 1/2*dt^2; 0.0 θ dt; 0.0 0.0 θ]
Bhat(θ) = B
Qhat(θ) = Q
Hhat(θ) = H
Rhat(θ) = R
# define a parametrized kalman filter
param_kf = ParamKalmanFilter(Ahat, Bhat, Qhat, Hhat, Rhat)


θ0 = 2.0
Ahat0 = Ahat(θ0)
kf0 = KalmanFilter(Ahat0, B, Q, H, R)
states0 = run_filter(kf, s0, action_sequence, sim_measurements)

loss = []
θ_range = 0.5:0.1:1.5
for θ_i in θ_range
    gs = gradient(𝛉 -> kf_likelihood(𝛉, param_kf, states0, action_sequence, sim_measurements), θ_i)
    l = kf_likelihood(θ_i, param_kf, states0, action_sequence, sim_measurements)
    push!(loss, l)
end

plot(θ_range, loss)
