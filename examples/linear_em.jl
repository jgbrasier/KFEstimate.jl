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

Ahat(θ) = [θ dt 1/2*dt^2; 0.0 θ dt; 0.0 0.0 θ]

θ0 = 2.0
Ahat0 = Ahat(θ0)
kf0 = KalmanFilter(Ahat0, B, Q, H, R)
states0 = run_filter(kf, s0, action_sequence, sim_measurements)


function kf_likelihood(θ, A, B, Q, H, R, state_beliefs::AbstractArray,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # drop initial s0 belief
    state_beliefs = state_beliefs[2:end]
    @assert length(state_beliefs) == length(measurement_history)
    N = length(measurement_history)
    # initialize log likelihood
    # l = o.R[1]
    l=0.0
    for (k, (s, y, u)) in enumerate(zip(state_beliefs, measurement_history, action_history))
        x_hat = A(θ)*s.x + B*u # predicted state prior
        P_hat = A(θ)*s.P*A(θ)' + Q # a priori state covariance
        v = y - H*x_hat # measurement pre fit residual
        S = H*P_hat*H' + R # pre fit residual covariance
        l += 1/2*(v'*inv(S)*v + log(det(S)))
    end
    return l
end

loss = []
θ_range = 0.5:0.1:1.5
for θ_i in θ_range
    gs = gradient(𝛉 -> kf_likelihood(𝛉, Ahat, B, Q, H, R, states0, action_sequence, sim_measurements), θ_i)
    l = kf_likelihood(θ_i, Ahat, B, Q, H, R, states0, action_sequence, sim_measurements)
    push!(loss, l)
end

plot(θ_range, loss)
