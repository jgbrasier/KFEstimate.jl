""" Linear Kalman Filter Structure """

abstract type AbstactFilter end

struct KalmanFilter <: AbstractFilter
    d::LinearDynamicsModel
    o::LinearObservationModel
end
