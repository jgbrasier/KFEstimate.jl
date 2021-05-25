""" Linear Kalman Filter Structure """

abstract type AbstractFilter end

struct KalmanFilter <: AbstractFilter
    d::LinearDynamicModel
    o::LinearObservationModel
end
