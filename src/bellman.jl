"""
    bellman(V, prob; upper_bound = false)

Compute robust Bellman update with the value function `V` and the interval probabilities `prob` 
that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed.

### Examples
```jldoctest
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

Vprev = collect(1:15)
Vcur = bellman(Vprev, prob; upper_bound = false)
```

!!! note
    This function will construct a workspace object and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.
    See [`construct_workspace`](@ref) for how to pre-allocate the workspace.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman(V, prob; upper_bound = false)
    Vres = similar(V, num_source(prob))
    return bellman!(Vres, V, prob; upper_bound = upper_bound)
end

"""
    bellman!(workspace, Vres, V, prob, stateptr; upper_bound = false, maximize = true)

Compute in-place robust Bellman update with the value function `V` and the interval probabilities
`prob` that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed. 

The output is constructed in the input `Vres` and returned. The workspace object is also modified,
and depending on the type, the policy cache may be modified as well.

### Examples

```jldoctest
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

V = collect(1:15)
workspace = construct_workspace(prob)
Vres = similar(V)

Vres = bellman!(workspace, Vres, V, prob; upper_bound = false, maximize = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman! end

function bellman!(Vres, V, prob; upper_bound = false)
    workspace = construct_workspace(prob)
    return bellman!(workspace, Vres, V, prob; upper_bound = upper_bound)
end

function bellman!(workspace, Vres, V, prob; upper_bound = false)
    return bellman!(workspace, Vres, V, prob, stateptr(prob); upper_bound = upper_bound)
end

##################
# Dense - simple #
##################
function bellman!(
    workspace::DenseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    l = lower(prob)
    g = gap(prob)

    # rev=true for maximization
    sortperm!(workspace.permutation, V; rev = upper_bound)
    
    @inbounds for jₛ in 1:(length(stateptr) - 1)
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        action_values = @view workspace.actions[1:(s₂ - s₁)]
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view l[:, jₐ]
            gapⱼ = @view g[:, jₐ]
            used = sum_lower(prob)[jₐ]

            action_values[i] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, used, workspace.permutation)
        end

        Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, V, jₛ, [], s₁, maximize)
    end

    return Vres
end

function bellman!(
    workspace::ThreadedDenseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    # rev=true for maximization
    sortperm!(workspace.permutation, V; rev = upper_bound)

    l = lower(prob)
    g = gap(prob)

    @inbounds @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        thread_actions = workspace.actions[tid]
        
        action_values = @view thread_actions[1:(s₂ - s₁)]
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view l[:, jₐ]
            gapⱼ = @view g[:, jₐ]
            used = sum_lower(prob)[jₐ]

            action_values[i] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, used, workspace.permutation)
        end

        Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, V, jₛ, [], s₁, maximize)
    end

    return Vres
end

function gap_value(V, gap::VR, sum_lower, perm) where {VR <: AbstractVector}
    remaining = 1.0 - sum_lower
    res = 0.0

    @inbounds for i in perm
        p = min(remaining, gap[i])
        res += p * V[i]

        remaining -= p
        if remaining <= 0.0
            break
        end
    end

    return res
end

###################
# Dense - product #
###################
function bellman!(
    workspace::DenseProductWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    l = lower(prob)
    g = gap(prob)

    @inbounds for other_index in eachotherindex(V)
        Vₒ = @view V[:, other_index...]
        perm = @view workspace.permutation[1:length(Vₒ)]

        # rev=true for maximization
        sortperm!(perm, Vₒ; rev = upper_bound)
        
        for jₛ in 1:(length(stateptr) - 1)
            s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
            action_values = @view workspace.actions[1:(s₂ - s₁)]
            for (i, jₐ) in enumerate(s₁:(s₂ - 1))
                lowerⱼ = @view l[:, jₐ]
                gapⱼ = @view g[:, jₐ]
                used = sum_lower(prob)[jₐ]

                action_values[i] = dot(Vₒ, lowerⱼ) + gap_value(Vₒ, gapⱼ, used, perm)
            end

            Vres[jₛ, other_index...] = extract_policy!(workspace.policy_cache, action_values, Vₒ, jₛ, other_index, s₁, maximize)
        end
    end

    return Vres
end

function bellman!(
    workspace::ThreadedDenseProductWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    @inbounds @threadstid tid for other_index in eachotherindex(V)
        ws = workspace.actions[tid]
        perm = @view ws.permutation[1:length(Vₒ)]

        Vₒ = @view V[:, other_index...]

        # rev=true for maximization
        sortperm!(perm, Vₒ; rev = upper_bound)

        l = lower(prob)
        g = gap(prob)

        for jₛ in 1:(length(stateptr) - 1)
            s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
            
            action_values = @view ws.actions[1:(s₂ - s₁)]
            for (i, jₐ) in enumerate(s₁:(s₂ - 1))
                lowerⱼ = @view l[:, jₐ]
                gapⱼ = @view g[:, jₐ]
                used = sum_lower(prob)[jₐ]

                action_values[i] = dot(Vₒ, lowerⱼ) + gap_value(Vₒ, gapⱼ, used, perm)
            end

            Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, Vₒ, jₛ, other_index, s₁, maximize)
        end
    end

    return Vres
end

###################
# Sparse - simple #
###################
function bellman!(workspace::ThreadedSparseWorkspace, Vres, V, prob, stateptr; upper_bound = false, maximize = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        ws = workspace.thread_workspaces[tid]
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        action_values = @view ws.actions[1:(s₂ - s₁)]

        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view l[:, jₐ]
            gapⱼ = @view g[:, jₐ]
            used = sum_lower(prob)[jₐ]

            Vp_workspace = @view ws.values_gaps[1:nnz(gapⱼ)]
            for (i, (V, p)) in
                enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                Vp_workspace[i] = (V, p)
            end

            # rev=true for maximization
            sort!(Vp_workspace; rev = upper_bound, by = first)

            action_values[i] = dot(V, lowerⱼ) + gap_value(Vp_workspace, used)
        end

        Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, V, jₛ, [], s₁, maximize)
    end

    return Vres
end

function bellman!(workspace::SparseWorkspace, Vres, V, prob, stateptr; upper_bound = false, maximize = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds for jₛ in 1:(length(stateptr) - 1)
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        action_values = @view workspace.actions[1:(s₂ - s₁)]
        
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view l[:, jₐ]
            gapⱼ = @view g[:, jₐ]
            used = sum_lower(prob)[jₐ]

            Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
            for (i, (V, p)) in
                enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                Vp_workspace[i] = (V, p)
            end

            # rev=true for maximization
            sort!(Vp_workspace; rev = upper_bound, by = first)

            action_values[i] = dot(V, lowerⱼ) + gap_value(Vp_workspace, used)
        end

        Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, V, jₛ, [], s₁, maximize)
    end

    return Vres
end

###################
# Sparse -product #
###################
function bellman!(workspace::ThreadedSparseProductWorkspace, Vres, V, prob, stateptr; upper_bound = false, maximize = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds @threadstid tid for other_index in eachotherindex(V)
        ws = workspace.thread_workspaces[tid]
        Vₒ = @view V[:, other_index...]
    
        for jₛ in 1:(length(stateptr) - 1)
            s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
            action_values = @view ws.actions[1:(s₂ - s₁)]

            for (i, jₐ) in enumerate(s₁:(s₂ - 1))
                lowerⱼ = @view l[:, jₐ]
                gapⱼ = @view g[:, jₐ]
                used = sum_lower(prob)[jₐ]

                Vp_workspace = @view ws.values_gaps[1:nnz(gapⱼ)]
                for (i, (V, p)) in
                    enumerate(zip(@view(Vₒ[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                    Vp_workspace[i] = (V, p)
                end

                # rev=true for maximization
                sort!(Vp_workspace; rev = upper_bound, by = first)

                action_values[i] = dot(Vₒ, lowerⱼ) + gap_value(Vp_workspace, used)
            end

            Vres[jₛ, other_index...] = extract_policy!(workspace.policy_cache, action_values, Vₒ, jₛ, other_index, s₁, maximize)
        end
    end

    return Vres
end

function bellman!(workspace::SparseProductWorkspace, Vres, V, prob, stateptr; upper_bound = false, maximize = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds for other_index in eachotherindex(V)
        Vₒ = @view V[:, other_index...]
        
        for jₛ in 1:(length(stateptr) - 1)
            s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
            action_values = @view workspace.actions[1:(s₂ - s₁)]

            for (i, jₐ) in enumerate(s₁:(s₂ - 1))
                lowerⱼ = @view l[:, jₐ]
                gapⱼ = @view g[:, jₐ]
                used = sum_lower(prob)[jₐ]

                Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
                for (i, (V, p)) in
                    enumerate(zip(@view(Vₒ[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                    Vp_workspace[i] = (V, p)
                end

                # rev=true for maximization
                sort!(Vp_workspace; rev = upper_bound, by = first)

                action_values[i] = dot(Vₒ, lowerⱼ) + gap_value(Vp_workspace, used)
            end

            Vres[jₛ] = extract_policy!(workspace.policy_cache, action_values, Vₒ, jₛ, [], s₁, maximize)
        end
    end

    return Vres
end

function gap_value(Vp, sum_lower)
    remaining = 1.0 - sum_lower
    res = 0.0

    @inbounds for (V, p) in Vp
        p = min(remaining, p)
        res += p * V

        remaining -= p
        if remaining <= 0.0
            break
        end
    end

    return res
end
