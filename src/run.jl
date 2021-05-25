
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


function run_filter(filter::AbstractFilter, R::AbstractMatrix, s0::State, action_history::Vector{A},
    measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}
    """
    """
    @assert length(action_history) == length(measurement_history)
    states = [s0]
    for (u, y) in zip(action_history, measurement_history)
        sp = prediction(filter.d, states[end])
        sn = correction(filter.o, R, sp, y)
        push!(states, sn)
    end
    return states
end

function unpack(states_history::Vector{<:State};
    dims::Vector{Int}=Vector{Int}())
    # set default to condense all dimensions
    if length(dims) == 0
        dims = collect(1:length(states_history[1].μ))
    end
    # set output sizes
    μ = zeros(typeof(states_history[1].μ[1]),
        length(states_history), length(dims))
    Σ = zeros(typeof(states_history[1].μ[1]),
        length(states_history), length(dims), length(dims))
    # iterate over belief_history and place elements appropriately
    for (i, state) in enumerate(states_history)
        μ[i,:] = state.s[dims]S
        Σ[i,:,:] = state.P[dims,dims]
    end
    return μ, Σ
end
