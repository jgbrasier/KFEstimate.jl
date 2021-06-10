using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux, Flux.Optimise
using Flux: Params, gradient
using ProgressBars
pathof(KFEstimate)

## pendulum simulation

dt = 0.001
g = 9.81

function f(x)
    dx = x
    dx[1] = x[1] + x[2]*dt
    dx[2] = x[2] - g*dt*sin(x[1])
    return dx
end

function h(x)
    return [sin(x[1]); 0.0]
end

Q = 0.1*[dt^3/3 dt^2/2; dt^2/2 dt]
R = 1.0*Matrix{Float64}(I, 2, 2)

ekf = ExtendedKalmanFilter(f, Q, h, R);
# run simulation
time_step = 0.0:10000
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
x = hcat(θ, dθ)
# plot
plot(time_step, x[2:end, 1], label = "simulated θ", legend=:bottomleft)
plot!(time_step, x[2:end, 2], label = "simulated dθ", legend=:bottomleft)
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
using Zygote, Statistics

fθ = Chain(Dense(2, 10, sigmoid), Dense(10, 2))
est_ekf = ExtendedKalmanFilter(fθ, Q, h, R); # Estimated Kalman FIlet
opt = ADAM(0.1)

function run_gradient(filter, action_history, measurement_history, Σ_history)
    s_grad = [State([0.0; 0.0], Matrix{Float64}(I, 2, 2))]
    # Ahats = [copy(filter.A)]
    # Bhats = [copy(filter.B)]
    @assert length(action_history) == length(measurement_history)
    for (u, y) in ProgressBar(zip(action_history, measurement_history))
        s = s_grad[end]
        for i in 1:20
            # print(x_grad, "\n")
            sp = prediction(filter, s, u)
            s = correction(filter, sp, y)
            grads = gradient(() -> step_loss(filter, s.x, s.P, u, y), Flux.Params(filter.f))
            println()
            for i in range(len())
            update!(opt, Flux.Params(filter.f), grads)
            println(Flux.Params(filter.f)[1].weight, "\n \n")
            # println(ϵx, ϵy, dμ, "\n")
        end
        break
        push!(s_grad, s)
        # push!(Ahats, copy(filter.A))
        # push!(Bhats, copy(filter.B))
    end
    return s_grad
end

grad_states = run_gradient(est_ekf, action_sequence, sim_measurements, Σ)
μgrad, Σgrad = unpack(grad_states)

l = @layout [a{0.7h};grid(1, 2)]
p1 = plot(time_step, [x[2:end, 1] x[2:end, 2]], label = ["simulated θ" "simulated dθ"], legend=:bottomright)
p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2]], label = ["filtered θ" "filtered dθ"], legend=:bottomright)
p1 = plot!(time_step, [μgrad[2:end, 1] μgrad[2:end, 2]], label = ["learned θ" "learned dθ"], legend=:bottomright)
# p2 = plot(time_step[250:end], L[250:end], title="A matrix loss")
p3 = plot(time_step, (x[2:end, :]-μ[2:end, :]).^2, title="KF vs. sim error")
p4 = plot(time_step[250:end], (x[251:end, :]-μgrad[251:end, :]).^2, title="grad vs. sim error")
plot(p1, p3, p4, layout=l, titlefont = font(12), size=(1000, 700))
xlabel!("time step (t)")


##

# wrap MLP into a an EKF dynamics function of parameters x and u

function step_loss(filter::ExtendedKalmanFilter, x, p, u, y)
    ϵx = x - filter.f(x)
    ϵy = y - filter.h(x)
    return -ϵx'*filter.R*ϵx + ϵy'*p*ϵy
end

fθ = Chain(Dense(2, 10, sigmoid), Dense(10, 2))
est_ekf = ExtendedKalmanFilter(fθ, Q, h, R)

s = State([0.0; 0.0], Matrix{Float64}(I, 2, 2))
u = action_sequence[1]
y = sim_measurements[1]
grads = gradient(() -> step_loss(est_ekf, s.x, s.P, u, y), Flux.Params(est_ekf.f))
update!(opt, Flux.Params(est_ekf.f), grads)
