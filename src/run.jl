
function run_simulation(filter::AbstractFilter, R::AbstractMatrix, s0::State, action_seq::AbstractArray)
    """ Simulate N=length(action_seq) points correspding to a
    Dynamic/Observation Model encapsulated in a KF struct.
    Start from an initial state belief s0"""
    meas = []
    states = [s0]
    for u in action_seq
        xp = dynamic(filter.d, states[end])
        ym = observation(filter.o, R, xp)
        push!(meas, y)
        push!(states, xp)
    end
    return states, meas
end


function run_filter(filter::AbstractFilter, R:AbstractMatrix, s0::State, action_history::Vector{A},
            measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    for u in action_history
        for (u, y) in zip(action_history, measurement_history)
            sp = prediction(filter.d, states[end])
            sn = correction(filter.o, R, sp, y)
            push!(states, sn)
    end
end
