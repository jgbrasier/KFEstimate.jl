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
using Flux, Flux.Optimise
using Statistics
using Distributions
import Random: rand

export
    AbstractFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    AbstractParamFilter,
    ParamKalmanFilter,
    ExtendedParamKalmanFilter,
    State
include("filters.jl")

export
    dynamic,
    param_dynamic,
    prediction,
    param_prediction,
    observation,
    param_observation,
    correction,
    param_correction,
    state_mse
include("ekf_functions.jl")
include("kf_functions.jl")

export
    kf_likelihood,
    ekf_likelihood
    online_kf_likihood,

include("likelihood.jl")

export
    run_simulation,
    run_filter,
    run_param_filter,
    run_kf_gradient,
    run_ekf_gradient,
    run_online_kf_gradient
include("run.jl")

export
    unpack
include("utils.jl")
end
