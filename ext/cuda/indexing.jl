function gpu_nextind(sizes::NTuple{N, <:Integer}, inds::NTuple{N, <:Integer}) where {N}
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