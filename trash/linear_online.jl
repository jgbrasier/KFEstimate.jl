using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux, Flux.Optimise
using ProgressBars
pathof(KFEstimate)
## Normal Kalman Filter

# dynamic model x = [p, v, a]
dt = 0.0001
A = [1.0 dt 1/2*dt^2; 0.0 1.0 dt; 0.0 0.0 1.0]
B = [0.0; 0.0; 1.0]
B = reshape(B, length(B), 1)
Q = 0.01*Matrix{Float64}(I, 3, 3)

# observation model, assume we can noisily measure position
H = randn(3, 3)
# H = [1 0 0; 0 1 0; 0 0 1]
R = 1.0*Matrix{Float64}(I, 3, 3)

kf = KalmanFilter(A, B, Q, H, R)

# run simulation
time_step = 0:1000
x0 = [0.0; 0.0; 0.0]
action_sequence = [[exp(-0.1*(t-1))] for t in time_step]
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
plot(time_step, [x[2:end, 1] x[2:end, 2] x[2:end, 3]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomleft)
plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtred p" "filtered v" "filtered a"], legend=:bottomleft)
xlabel!("time step (t)")

## Process matrix parameter estimation
# for each state k, perform gradient descent on process log-likelihood using autodiff

using Zygote, Statistics

Aest = randn(3, 3)
Best = randn(3, 1)

est_kf = KalmanFilter(Aest, Best, Q, H, R)
opt = ADAM(0.00002)

function run_gradient(filter::KalmanFilter, action_history, measurement_history)
    s_grad = [State([0.0; 0.0; 0.0], Matrix{Float64}(I, 3, 3))]
    # Ahats = [copy(filter.A)]
    # Bhats = [copy(filter.B)]
    # η = 0.00002
    @assert length(action_history) == length(measurement_history)
    L = []
    for (u, y) in ProgressBar(zip(action_history, measurement_history))
        old_A = copy(filter.A)
        s = s_grad[end]
        # s_old = s_grad[end]
        for i in 1:400
            # print(x_grad, "\n")
            s = prediction(filter, s, u)
            s = correction(filter, s, y)
            dA, = gradient(() -> mse_loss(filter, s, u, y), Params([filter.A]))
            # filter.A += η*dA
            update!(opt, filter.A, dA)
            dB, = gradient(() -> mse_loss(filter, s, u, y), Params([filter.B]))
            update!(opt, filter.B, dB)
            # println(ϵx, ϵy, dμ, "\n")
        end
        new_A = copy(filter.A)
        push!(L, norm(old_A-new_A))
        push!(s_grad, s)
        # push!(Ahats, copy(filter.A))
        # push!(Bhats, copy(filter.B))
    end
    return s_grad, L
end

grad_states, L = run_gradient(est_kf, action_sequence, sim_measurements)
μgrad, Σgrad = unpack(grad_states)

l = @layout [a{0.7h};grid(1, 3)]
p1 = plot(time_step, [x[2:end, 1] x[2:end, 2] x[2:end, 3]], label = ["simulated p" "simulated v" "simulated a"], legend=:bottomright)
p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2] μ[2:end, 3]], label = ["filtered p" "filtered v" "filtered a"], legend=:bottomright)
p1 = plot!(time_step, [μgrad[2:end, 1] μgrad[2:end, 2] μgrad[2:end, 3]], label = ["learned p" "learned v" "learned a"], legend=:bottomright)
p2 = plot(time_step[250:end], L[250:end], title="A matrix loss")
p3 = plot(time_step, (x[2:end, :]-μ[2:end, :]).^2, title="KF vs. sim error")
p4 = plot(time_step[250:end], (x[251:end, :]-μgrad[251:end, :]).^2, title="grad vs. sim error")
plot(p1, p2, p3, p4, layout=l, titlefont = font(12), size=(1000, 700))
xlabel!("time step (t)")
