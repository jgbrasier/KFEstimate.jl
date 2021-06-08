using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux.Optimise
using ProgressBars
pathof(KFEstimate)
## Normal Kalman Filter

# dynamic model x = [p, v]
dt = 0.001
A = [1.0 dt 1/2*dt^2; 0.0 1.0 dt; 0.0 0.0 1.0]
B = [0.0; 0.0; 1.0]
B = reshape(B, length(B), 1)
Q = 0.01*Matrix{Float64}(I, 3, 3)

# observation model, assume we can noisily measure position
H = randn(3, 3)
R = 10.0*Matrix{Float64}(I, 3, 3)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0.0:1000
x0 = [0.0; 0.0; 0.0]
action_sequence = [[exp(-0.01*t)] for t in time_step]
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

Aest = randn(3, 3)

est_kf = KalmanFilter(Aest, B, Q, H, R)
function run_gradient(filter, action_history, measurement_history, Σ_history)
    x_grads = [[0.0; 0.0; 0.0]]
    p_grad = Matrix{Float64}(I, 3, 3)
    Ahats = [copy(filter.A)]
    @assert length(action_history) == length(measurement_history)
    for (u, y, p) in ProgressBar(zip(action_history, measurement_history, Σ_history))
        x_grad = x_grads[end]
        Ahat = Ahats[end]
        for i in 1:20
            ϵx = x_grad - (Ahat*x_grad + filter.B*u)
            ϵy = y - filter.H*x_grad
            dμ = p_grad*ϵx - filter.H'*filter.R*ϵy
            x_grad -= (0.05*dμ)
            Ahat_grad = p*ϵx*x_grad'

            Ahat += 0.0000005*Ahat_grad
        end
        push!(x_grads, x_grad)
        push!(Ahats, Ahat)
    end
    return x_grads
end

grad_states = run_gradient(est_kf, action_sequence, sim_measurements, Σ)

pgrad = [x[1] for x in grad_states]
vgrad = [x[2] for x in grad_states]
agrad = [x[3] for x in grad_states]

plot(time_step, [p[2:end] v[2:end] a[2:end]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomright)
plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtered p" "filtered v" "filtered a"], legend=:bottomright)
plot!(time_step, [pgrad[2:end] vgrad[2:end] agrad[2:end]], label = ["GD p" "GD v" "GD a"], legend=:bottomright)
xlabel!("time step (t)")
