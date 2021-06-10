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
using Statistics
using Distributions
import Random: rand

export
    AbstractFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    State
include("filters.jl")

export
    dynamic,
    prediction,
    observation,
    correction,
    pre_fit
include("ekf_functions.jl")
include("kf_functions.jl")

export
    run_simulation,
    run_filter,
    run_noise_estimation
include("run.jl")

export
    unpack,
    unpack_history,
    likelihood,
    compute_noise_loss
include("utils.jl")
end
