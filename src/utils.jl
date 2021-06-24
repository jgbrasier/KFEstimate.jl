function unpack(states_history::Vector{<:State};
    dims::Vector{Int}=Vector{Int}())
    # set default to condense all dimensions
    if length(dims) == 0
        dims = collect(1:length(states_history[1].x))
    end
    # set output sizes
    μ = zeros(typeof(states_history[1].x[1]),
        length(states_history), length(dims))
    Σ = zeros(typeof(states_history[1].x[1]),
        length(states_history), length(dims), length(dims))
    # iterate over belief_history and place elements appropriately
    for (i, state) in enumerate(states_history)
        μ[i,:] = state.x[dims]
        Σ[i,:,:] = state.P[dims,dims]
    end
    return μ, Σ
end
