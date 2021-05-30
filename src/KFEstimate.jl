__precompile__(true)
module KFEstimate

# Write your package code here.
using LinearAlgebra
using ForwardDiff
using Zygote
using Random
using Revise
using Plots
using ProgressBars
using Flux.Optimise
import Random: rand

export
    AbstractFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    State
include("filters.jl")

export
    dynamic,
    predict,
    observation,
    correction,
    pre_fit
include("kf_functions.jl")
include("ekf_functions.jl")

export
    run_simulation,
    run_filter,
    run_linear_estimation
include("run.jl")

export
    unpack,
    likelihood,
    compute_noise_loss
include("utils.jl")
end
