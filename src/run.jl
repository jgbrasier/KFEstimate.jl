
function simulate(filter::AbstractFilter, R::AbstractMatrix, s0::State action_seq::AbstractArray)
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
