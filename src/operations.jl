#=
operations.jl; This file implements operations that can be done on hypervectors to enable them to encode text-based data.
=#

# Remark: use element-wise reduce, maybe using LazyArrays?


#=

| Operation            | symbol | remark                                                                                                          |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Bundling/aggregating | `+`    | combines the information of two vectors into a new vector similar to both                                       |
| Binding              | `*`    | mapping, combines the two vectors in something different from both, preserves distance, distributes of bundling |
| Shifting             | `ρ`    | Permutation (in practice cyclic shifting), distributes over addition, conserves distance                        |
=#

"""
    grad2bipol(x::Number)

Maps a graded number in [0, 1] to the [-1, 1] interval.
"""
grad2bipol(x::Real) = 2x - one(x)


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Real) = (x + one(x)) / 2

three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x)-x) * y + x * (one(y)-y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))  # currently just *

aggfun(::Type{<:AbstractHV}) = +
aggfun(::GradedHV) = three_pi
aggfun(::GradedBipolarHV) = three_pi_bipol

bindfun(::AbstractHV) = *
bindfun(::BinaryHV) = ⊻
bindfun(::GradedHV) = fuzzy_xor
bindfun(::GradedBipolarHV) = fuzzy_xor_bipol

neutralbind(hdv::AbstractHV) = one(eltype(hdv))
neutralbind(hdv::BinaryHV) = false
neutralbind(hdv::GradedHV) = zero(eltype(hdv))
neutralbind(hdv::GradedBipolarHV) = -one(eltype(hdv))

noisy_and(a,b) = a==b ? a : rand(Bool)

function elementreduce!(f, itr, init)
    return foldl(itr; init) do acc, value
        acc .= f.(acc, value)
    end
end

# computes `r[i] = f(x[i], y[i+offset])`
# assumes postive offset (for now)
@inline function offsetcombine!(r, f, x, y, offset=0)
    @assert length(r) == length(x) == length(y)
    n = length(r)
    if offset==0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

@inline function offsetcombine(f, x::V, y::V, offset=0) where {V<:AbstractVecOrMat}
    @assert length(x) == length(y)
    r = similar(x)
    n = length(r)
    if offset==0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

# AGGREGATION
# -----------

# binary and bipolar: use majority
function aggregate(hvr::Union{BinaryHV,BipolarHV}, hdvs, r)
    m = length(hdvs)
    for hv in hdvs
        r .+= hv.v
    end
    if iseven(m)  # break ties
        r .+= bitrand(length(r))
    end
    hvr = similar(hvr)
    hvr.v .= r.>m/2
    return hvr
end
   
# ternary: just add them, no normalization by default
function aggregate(::TernaryHV, hdvs, r;
            normalize=false)
    for hv in hdvs
        r .+= hv.v
    end
    normalize && clamp!(r, -1, 1)
    return TernaryHV(r)
end

# realhv: just add + rescale with sqrt m
function aggregate(::RealHV, hdvs, r)
    m = 0
    for hv in hdvs
        r .+= hv.v
        m += 1
    end
    r ./= sqrt(m)
    return RealHV(r)
end

function aggregate(::GradedHV, hdvs, r)
    for hv in hdvs
        r .= three_pi.(r, hv.v)
    end
    return GradedHV(r)
end

function aggregate(::GradedBipolarHV, hdvs, r)
    for hv in hdvs
        r .= three_pi_bipol.(r, hv.v)
    end
    return GradedBipolarHV(r)
end

function aggregate(hdvs; kwargs...)
    hv = first(hdvs)
    r = empty_vector(hv)
    return aggregate(hv, hdvs, r, kwargs...)
end

Base.:+(hv1::HV, hv2::HV) where {HV<:AbstractHV} = aggregate((hv1, hv2))

#=
aggregate(hdvs::AbstractVector{<:AbstractHDV}, args...; kwargs...) = aggregate!(similar(first(hdvs)), hdvs, args...; kwargs...)
aggregate(hdvs::NTuple{N,T}, args...; kwargs...) where {N,T<:AbstractHDV} = aggregate!(similar(first(hdvs)), hdvs, args...; kwargs...)

Base.:+(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = aggregate!(similar(hdv1), (hdv1, hdv2))

clearhdv!(r::AbstractHDV) = fill!(r.v, zero(eltype(r)))
clearhdv!(r::GradedHDV) = fill!(r.v, one(eltype(r))/2)

function aggregate!(r::AbstractHDV, hdvs; clear=true, norm=false)
    clear && clearhdv!(r)
    aggr = aggfun(r)
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, aggr, acc, value.v, value.offset)
    end
    norm && normalize!(r)
    return r
end

function aggregate!(r::AbstractHDV, hdvs, weights; clear=true, norm=false)
    @assert length(hdvs) == length(weights) "You have to provide the same number of weights as vectors."
    clear && clearhdv!(r)
    aggr = aggfun(r)
    foldl(zip(hdvs, weights), init=r.v) do acc, (value, weight)
        offsetcombine!(acc, aggr, acc, weight .* value.v, value.offset)
    end
    for (hdv, weight) in zip(hdvs, weights)
        r.m += weight * hdv.m
    end
    norm && normalize!(r)
    return r
end

aggregatewith!(r::AbstractHDV, hdvs; kwargs...) = aggregate!(r, hdvs; clear=false, kwargs...)
=#

# BINDING
# -------

Base.bind(hv1::BinaryHV, hv2::BinaryHV) = BinaryHV(hv1.v .⊻ hv2.v)
Base.bind(hv1::BipolarHV, hv2::BipolarHV) = BipolarHV(hv1.v .⊻ hv2.v)
Base.bind(hv1::TernaryHV, hv2::TernaryHV) = TernaryHV(hv1.v .* hv2.v)
Base.bind(hv1::RealHV, hv2::RealHV) = RealHV(hv1.v .* hv2.v)
Base.bind(hv1::GradedHV, hv2::GradedHV) = GradedHV(fuzzy_xor.(hv1.v, hv2.v))
Base.bind(hv1::GradedBipolarHV, hv2::GradedBipolarHV) = GradedBipolarHV(fuzzy_xor_bipol.(hv1.v, hv2.v))

Base.:*(hv1::HV, hv2::HV) where {HV<:AbstractHV} = bind(hv1, hv2)

#=
Base.bind(hdvs::AbstractVector{<:AbstractHDV}) = bind!(similar(first(hdvs)), hdvs)
Base.bind(hdvs::NTuple{N,T}) where {N,T<:AbstractHDV} = bind!(similar(first(hdvs)), hdvs)

Base.:*(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = bind!(similar(hdv1), (hdv1, hdv2))

function bind!(r::AbstractHDV, hdvs)
    fill!(r.v, neutralbind(r))
    r.m = 1  # fresh vector
    # extract the normalizer
    nr = normalizer(r)
    binder = (x, y) -> bindfun(r)(nr(x), nr(y))
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, binder, acc, value.v, value.offset)
    end
    return r
end
=#

# SHIFTING
# --------

shift!(hv::AbstractHV, k=1) = circshift!(hv.v, k)

function shift(hv::AbstractHV, k=1)
    r = similar(hv)
    r.v .=  circshift(hv.v, k)
    return r
end

function shift!(hv::V, k=1) where {V<:Union{BinaryHV,BipolarHV}}
    v = similar(hv.v)  # empty bitvector
    hv.v .= circshift!(v, hv.v, k)
    return hv
end

function shift(hv::V, k=1) where {V<:Union{BinaryHV,BipolarHV}}
    v = similar(hv.v)  # empty bitvector
    return V(circshift!(v, hv.v, k))
end

ρ(hv::AbstractHV, k=1) = shift(hv, k)
ρ!(hv::AbstractHV, k=1) = shift!(hv, k)


# COMPARISON
# ----------

Base.isequal(v::AbstractHV, u::AbstractHV) = v.v == u.v


"""
    Base.isapprox(u::AbstractHV, v::AbstractHV, atol=length(u)/100, ptol=0.01)

Measurures when two hypervectors are similar (have more elements in common than expected
by chance).

One can specify either:
- `atol=N/100` number of matches more than due to chance needed for being assumed similar
- `ptol=0.01` threshold for seeing that many matches due to chance
"""
function Base.isapprox(u::T, v::T; atol=length(u)/100, ptol=0.01) where T<:Union{BinaryHV,BipolarHV}
    @assert length(u) == length(v) "Vectors have to be of equal length"
    N = length(u)
    missmatches = sum(ui!=vi for (ui, vi) in zip(u, v))
    matches = N - missmatches
    # probability of seeing fewer mismatches due to chance
    pval = cdf(Binomial(N, 0.5), missmatches)
    return pval < ptol || matches - N/2 > atol
end

"""
    Base.isapprox(u::AbstractHV, v::AbstractHV, atol=length(u)/100, ptol=0.01)

Measurures when two hypervectors are similar (have more elements in common than expected
by chance) using the Hamming distance. Uses a bootstrap to construct a null distribution.

One can specify either:
- `ptol=1e-10` threshold for seeing that many matches due to chance
- `N_bootstap=200` number of samples for bootstrapping
"""
function Base.isapprox(u::T, v::T; ptol=1e-10, N_bootstrap=500) where T<:AbstractHV
    @assert length(u) == length(v) "Vectors have to be of equal length"
    N = length(u)
    # bootstrap to find the zero distr
    B = [abs(rand(u) - rand(v)) for _ in 1:N_bootstrap]
    Bmean = N * mean(B)
    Bstd = sqrt(N) * std(B)
    # Hamming distance
    d = sum(abs(ui-vi) for (ui, vi) in zip(u, v))
    # probability of seeing fewer mismatches due to chance
    pval = cdf(Normal(Bmean, Bstd), d)
    return pval < ptol
end


# PERTURBATION
# ------------

function randbv(n::Int, m::Int)
    v = falses(n)  # empty vector
    v[1:m] .= true # set first m elements to 1
    return shuffle!(v)
end

function randbv(n::Int, p::Number)
    @assert 0 ≤ p ≤ 1 "p should be a valid probability"
    return randbv(n, round(Int, p*n))
end

function randbv(n::Int, I)
    v = falses(n)
    v[I] .= true
    return v
end


function perturbate!(::Type{HVByteVec}, hv::HV, I, dist=eldist(hv)) where {HV<:AbstractHV}
    hv.v[I] .= rand(dist, length(I))
    hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, M::BitVector, dist=eldist(hv)) where {HV<:AbstractHV}
    hv.v[M] .= rand(dist, sum(M))
    hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, p::Number, args...) where {HV<:AbstractHV}
    perturbate!(hv, randbv(length(hv), p), args...)
end

function perturbate!(::Type{HVBitVec}, hv::AbstractHV, binargs)
    n = length(hv)
    M = randbv(n, binargs)  # turn whatever into a mask
    hv.v .⊻= M
    return hv
end

perturbate!(hv, args...) = perturbate!(vectype(hv), hv, args...)

perturbate(hv::AbstractHV, args...; kwargs...) = perturbate!(copy(hv), args...; kwargs...)