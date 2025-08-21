#=
types.jl

Implements the basic types for the different hypervectors (wrappers for ordinary vectors)

Contains:
- BinaryHV
- BipolarHV
- TernaryHV
- RealHV
- GradedHV
- GradedBipolarHV

Every hypervector HV has the following basic functionality
- random generation using the Constructor ()
- norm/sum/normalize...

TODO: 
- [ ] SparseHV
- [ ] support for different types
- [ ] complex HDC
=#

abstract type AbstractHV{T} <: AbstractVector{T} end

#Base.collect(hv::AbstractHV) = hv.v
Base.sum(hv::AbstractHV) = sum(hv.v)
Base.size(hv::AbstractHV) = size(hv.v)
Base.getindex(hv::AbstractHV, i) = hv.v[i]
Base.similar(hv::T) where {T<: AbstractHV} = T(length(hv)) 
LinearAlgebra.norm(hv::AbstractHV) = norm(hv.v)
LinearAlgebra.normalize!(hv::AbstractHV) = hv
Base.hash(hv::AbstractHV) = hash(hv.v)
Base.copy(hv::HV) where {HV<:AbstractHV} = HV(copy(hv.v))


get_vector(v::AbstractVector) = v
get_vector(hv::AbstractHV) = hv.v

# Gives an empty Vector (filled with neutral elelment) that
# the `hv::AbstractHV` type uses.
empty_vector(hv::AbstractHV) = zero(hv.v)

eldist(hv::AbstractHV) = eldist(typeof(hv))

# trait for checking which vector is used internall


# We always provide a constructor with optinal dimensionality (n=10,000 by default) and
# a method `similar`.

# BipolarHV
# ---------

struct BipolarHV <: AbstractHV{Int}
    v::BitVector
    BipolarHV(v::BitVector) = new(v)
end

BipolarHV(n::Integer=10_000) = BipolarHV(bitrand(n))
BipolarHV(v::AbstractVector) = BipolarHV(v .> 0)

Base.getindex(hv::BipolarHV, i) = hv.v[i] ? 1 : -1
Base.sum(hv::BipolarHV) = 2sum(hv.v) - length(hv.v)
LinearAlgebra.norm(hv::BipolarHV) = sqrt(length(hv))

# needed for aggregation
empty_vector(hv::BipolarHV) = zeros(Int, length(hv))

eldist(::Type{BipolarHV}) = 2Bernoulli(0.5) - 1

# TernaryHV
# ---------

struct TernaryHV <: AbstractHV{Int}
    v::Vector{Int}
end

TernaryHV(n::Int=10_000) = TernaryHV(rand((-1, 1), n))

function LinearAlgebra.normalize!(hv::TernaryHV)
    clamp!(hv.v, -1, 1)
    return hv
end

LinearAlgebra.normalize(hv::TernaryHV) = TernaryHV(clamp.(hv, -1, 1))

eldist(::Type{TernaryHV}) = 2Bernoulli(0.5) - 1


# `BinaryHV` contain binary vectors.
# ---------

struct BinaryHV <: AbstractHV{Bool}
    v::BitVector
end

BinaryHV(n::Integer=10_000) = BinaryHV(bitrand(n))
BinaryHV(v::AbstractVector{Bool}) = BinaryHV(BitVector(v))

# needed for aggregation
empty_vector(hv::BinaryHV) = zeros(Int, length(hv))

eldist(::Type{BinaryHV}) = Bernoulli(0.5)

# `RealHV` contain real numbers, drawn from a distribution
# --------

struct RealHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
end

RealHV(n::Integer=10_000, distr::Distribution=eldist(RealHV)) = RealHV(rand(distr,n))


Base.similar(hv::RealHV) = RealHV(length(hv), eldist(RealHV)) 

function normalize!(hv::RealHV)
    hv.v .*= std(hv.distr) / std(hv.v)
    return hv
end

eldist(::Type{<:RealHV}) = Normal()


# GradedHV are vectors in $[0, 1]^n$, allowing for graded relations.
# ----------------

struct GradedHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
    #GradedHV(v::AbstractVector{T}) where {T<:Real} = new{T}(clamp!(v,0,1))
end

function GradedHV(n::Int=10_000, distr=eldist(GradedHV))
    @assert 0 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [0,1]"
    GradedHV(rand(distr, n))
end

Base.similar(hv::GradedHV) = GradedHV(length(hv), eldist(GradedHV))

# distribution used for sampling graded HVs
eldist(::Type{<:GradedHV}) = Beta(1, 1)

# neutral element of a graded HV is 0.5
empty_vector(hv::GradedHV) = fill!(zero(hv.v), 0.5)

LinearAlgebra.normalize!(hv::GradedHV) = clamp!(hv.v, 0, 1)

function Base.zeros(hv::GradedHV)
    v = similar(hv.v)
    return fill!(v, one(eltype(v))/2)
end

# GradedBipolarHV are vectors in $[-1, 1]^n$, allowing for graded relations.
# ---------------



struct GradedBipolarHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
    #GradedBipolarHV(v::AbstractVector{T}) where {T<:Real} = new{T}(clamp!(v,-1,1))
end

function GradedBipolarHV(n::Int=10_000, distr::Distribution=eldist(GradedBipolarHV))
    @assert -1 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [-1,1]"
    GradedBipolarHV(rand(distr, n))
end

# distribution used for sampling graded bipolar HVs
eldist(::Type{<:GradedBipolarHV}) = 2eldist(GradedHV) - 1

#GradedBipolarHV(n::Int) = GradedBipolarHV(graded_bipol_distr, n)

Base.similar(hv::GradedBipolarHV) = GradedBipolarHV(length(hv))
LinearAlgebra.normalize!(hv::GradedBipolarHV) = clamp!(hv.v, -1, 1)


# TRAITS
# ------

abstract type HVTraits end

struct HVByteVec <: HVTraits end

struct HVBitVec <: HVTraits end

vectype(::AbstractHV) = HVByteVec
vectype(::BinaryHV) = HVBitVec
vectype(::BipolarHV) = HVBitVec