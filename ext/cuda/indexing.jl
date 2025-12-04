Base.@propagate_inbounds function gpu_nextind(
    sizes::NTuple{N, Int32},
    inds::NTuple{N, Int32},
) where {N}
    inds_new = inds
    reset_last = false
    for d in Base.OneTo(N)
        if inds_new[d] < sizes[d]
            inds_new = Base.setindex(inds_new, inds_new[d] + Int32(1), d)
            break
        else
            inds_new = Base.setindex(inds_new, Int32(1), d)
            reset_last = (d == N)
        end
    end
    return inds_new, reset_last
end

Base.@propagate_inbounds function sub2ind_gpu(
    sizes::NTuple{N, Int32},
    inds::NTuple{N, Int32},
) where {N}
    ind = zero(Int32)

    for i in StepRange(N, -1, 1)
        ind *= sizes[i]
        ind += inds[i] - one(Int32)
    end

    return ind + one(Int32)
end

Base.@propagate_inbounds function ind2sub_gpu(sizes::NTuple{N, Int32}, ind::Int32) where {N}
    inds = ntuple(_ -> zero(Int32), N)
    ind -= one(Int32) # adjust for 1-based indexing

    for d in eachindex(sizes)
        assume(ind >= one(Int32))
        assume(sizes[d] >= one(Int32))

        ind, indsub = divrem(ind, sizes[d])
        inds = Base.setindex(inds, indsub + one(Int32), d)
    end

    return inds
end
