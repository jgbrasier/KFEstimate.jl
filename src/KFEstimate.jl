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
    State
include("filters.jl")

export
    dynamic,
    prediction,
    observation,
    correction,
    pre_fit,
    state_mse
include("ekf_functions.jl")
include("kf_functions.jl")

export
    run_simulation,
    run_filter
include("run.jl")

export
    unpack
include("utils.jl")
end
