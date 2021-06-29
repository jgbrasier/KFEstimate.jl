using Revise, KFEstimate
using LinearAlgebra, Plots
using Flux, Flux.Optimise
using Flux: Params, gradient
using ProgressBars
pathof(KFEstimate)

## pendulum simulation

global dt = 0.001
g = 9.81

function f(x)
    dx = x
    dx[1] = x[1] + x[2]*dt
    dx[2] = x[2] - g*dt*sin(x[1])
    return dx
end

function h(x)
    return [sin(x[1]); sin(x[2])]
end

Q = 0.01*[dt^3/3 dt^2/2; dt^2/2 dt]
R = 1.0*Matrix{Float64}(I, 2, 2)

ekf = ExtendedKalmanFilter(f, Q, h, R);
# run simulation
time_step = 0.0:5000
x0 = [0.0, 1.0]
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
plot!(time_step, μ[2:end, 1], label = "measured θ", legend=:bottomleft)
plot!(time_step, μ[2:end, 2], label = "estimated dθ", legend=:bottomleft)
xlabel!("time step (t)")
##
# using Zygote, Statistics
#
# hidden_dim = 24
# fθ = Chain(Dense(length(x0), hidden_dim, sigmoid), Dense(hidden_dim, length(x0)))
# est_ekf = ExtendedKalmanFilter(fθ, Q, h, R); # Estimated Kalman FIlet
# opt = ADAM(0.00001)
#
#
# function run_gradient(filter, action_history, measurement_history)
#     s_grad = []
#     # Ahats = [copy(filter.A)]
#     # Bhats = [copy(filter.B)]
#     @assert length(action_history) == length(measurement_history)
#     for i in  ProgressBar(1:10)
#         s_grad = [State([0.0; 0.0], Matrix{Float64}(I, 2, 2))]
#         for (u, y) in zip(action_history, measurement_history)
#             s = s_grad[end]
#             sp = prediction(filter, s, u)
#             s = correction(filter, sp, y)
#             ps = Flux.params(filter.f)
#             grads = gradient(ps) do
#                 likelihood(filter, s, u, y)
#             end
#             update!(opt, ps, grads)
#             # println(grads[ps[1]])
#             # println(filter.f[1].weight, "\n")
#             # println(ϵx, ϵy, dμ, "\n")
#             push!(s_grad, s)
#         end
#         # break
#         # push!(Ahats, copy(filter.A))
#         # push!(Bhats, copy(filter.B))
#     end
#     return s_grad
# end
#
# grad_states = run_gradient(est_ekf, action_sequence, sim_measurements)
# μgrad, Σgrad = unpack(grad_states)
#
# l = @layout [a{0.7h};grid(1, 2)]
# p1 = plot(time_step, [θ[2:end, 1] θ[2:end, 2]], label = ["simulated θ" "simulated dθ"], legend=:bottomright, title="state trajectories")
# p1 = plot!(time_step, [μ[2:end, 1] μ[2:end, 2]], label = ["filtered θ" "filtered dθ"], legend=:bottomright, title="state trajectories")
# p1 = plot!(time_step, [μgrad[2:end, 1] μgrad[2:end, 2]], label = ["learned θ" "learned dθ"], legend=:bottomright, title="state trajectories")
# # p2 = plot(time_step[250:end], L[250:end], title="A matrix loss")
# p3 = plot(time_step, (θ[2:end, :]-μ[2:end, :]).^2, title="KF vs. sim error")
# p4 = plot(time_step[250:end], (θ[251:end, :]-μgrad[251:end, :]).^2, title="grad vs. sim error")
# plot(p1, p3, p4, layout=l, titlefont = font(12), size=(1000, 700))
# xlabel!("time step (t)")
