""" Linear Kalman Filter Structure """

abstract type AbstractFilter end

struct KalmanFilter <: AbstractFilter
    d::LinearDynamicModel
    o::LinearObservationModel
end

struct ExtendedKalmanFilter <: AbstractFilter
    d::NonLinearDynamicModel
    o::NonLinearObservationModel
end
