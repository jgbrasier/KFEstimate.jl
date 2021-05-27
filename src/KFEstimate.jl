__precompile__(true)
module KFEstimate

# Write your package code here.
using LinearAlgebra
using ForwardDiff
using Zygote
using Random
using Revise
using Plots
import Random: rand

export
    State,
    DynamicModel,
    LinearDynamicModel,
    ObservationModel,
    LinearObservationModel
include("models.jl")

export
    dynamic,
    predict,
    observation,
    correction,
    pre_fit
include("kf_functions.jl")

export
    AbstractFilter,
    KalmanFilter
include("filters.jl")

export
    run_simulation,
    run_filter,
    unpack
include("run.jl")

export
    unpack,
    likelihood,
    compute_loss
include("utils.jl")
end
