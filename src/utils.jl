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

function likelihood(filter::AbstractFilter, state_beliefs::AbstractArray,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # drop initial s0 belief
    state_beliefs = state_beliefs[2:end]
    @assert length(state_beliefs) == length(measurement_history)
    N = length(measurement_history)
    # initialize log likelihood
    # l = o.R[1]
    l=0.0
    for (k, (s, y, u)) in enumerate(zip(state_beliefs, measurement_history, action_history))
        l += 1/2*pre_fit(filter, s, u, y)
    end
    return l
end

function compute_noise_loss(filter::AbstractFilter, r_range::StepRangeLen, s0::State,
    action_history::AbstractArray, measurement_history::AbstractArray)
    # compute loss (log-likelihood) for a given range of noise covariances
    loss = []
    @assert length(action_history) == length(measurement_history)
    for r in r_range
        R = Symmetric(r*Matrix{Float64}(I, 1, 1))
        filter.R = R
        filtered_states = run_filter(filter, s0, action_history, measurement_history)
        l = likelihood(filter, filtered_states, action_history, measurement_history)
        push!(loss, l)
    end
    return loss
end
