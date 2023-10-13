
function IMDP.probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::AbstractStateOrdering,
    indices,
) where {R, MR <: CuMatrix{R}}
    copyto!(p, lower(prob))

    n = size(p, 2)

    threads = 1024
    blocks = ceil(Int64, n / threads)

    @cuda blocks=blocks threads=threads probability_assignment_from_kernel!(p, gap(prob), sum_lower(prob), indices, ordering)

    return p
end

function probability_assignment_from_kernel!(
    p::MR,
    gap::MR,
    sum_lower::VR,
    indices::CuVector{Int64},
    ordering,
) where {R, VR <: CuVector{R}, MR <: CuMatrix{R}}

    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if k <= length(indices)
        j = indices[k]
        p = view(p, :, j)
        g = view(gap, :, j)

        remaining = 1 - sum_lower[j]

        for i in perm(ordering, j)
            @inbounds p[i] += g[i]
            @inbounds remaining -= g[i]

            if remaining < 0.0
                @inbounds p[i] += remaining
                remaining = 0.0
                break
            end
        end
    end

    return p
end
