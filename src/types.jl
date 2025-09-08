# Types for Hyperdimensional Computing
#
# This file implements the core hypervector types and their interfaces.
# All types support the fundamental HDC operations: bundling, binding, and permutation.
#
# TODO: 
# - [ ] SparseHV
# - [ ] ComplexHV  
# - [ ] Better type parameter handling

"""
    AbstractHV{T} <: AbstractVector{T}

Abstract supertype for all hyperdimensional vectors (hypervectors).

Hyperdimensional vectors are high-dimensional vectors (typically 10,000+ dimensions) used in 
hyperdimensional computing (HDC) for representing and manipulating symbolic information. All 
concrete hypervector types support the fundamental HDC operations:

- **Bundling/Aggregation**: Combining multiple vectors into a single similar vector
- **Binding**: Creating a vector dissimilar to its inputs that can be reversed
- **Permutation**: Reordering elements to encode position or sequence information

# Interface
All subtypes have the following functionality:
- A default constructor taking optional dimension `n::Integer`
- Vector operations via `AbstractVector` interface
- Support for `bundle`, `bind`, and `shift` operations

# See also
[`BinaryHV`](@ref), [`BipolarHV`](@ref), [`RealHV`](@ref), [`GradedHV`](@ref), [`TernaryHV`](@ref)
"""
abstract type AbstractHV{T} <: AbstractVector{T} end

# ============================================================================
# AbstractHV Interface Implementation
# ============================================================================

# Standard AbstractVector interface
Base.sum(hv::AbstractHV) = sum(hv.v)
Base.size(hv::AbstractHV) = size(hv.v)
Base.getindex(hv::AbstractHV, i) = hv.v[i]
Base.similar(hv::T) where {T<:AbstractHV} = T(length(hv))
LinearAlgebra.norm(hv::AbstractHV) = norm(hv.v)
LinearAlgebra.normalize!(hv::AbstractHV) = hv  # Default: no-op, overridden by subtypes
Base.hash(hv::AbstractHV) = hash(hv.v)
Base.copy(hv::HV) where {HV<:AbstractHV} = HV(copy(hv.v))

# Utility functions

"""
    get_vector(hv::AbstractHV)
    get_vector(v::AbstractVector)

Extract the underlying vector from a hypervector or pass through regular vectors.

Utility function for generic code that needs to access the raw vector data.

# Examples
```julia
julia> hv = BipolarHV(5); 
julia> get_vector(hv) isa BitVector
true

julia> v = [1,2,3]; get_vector(v) === v
true
```
"""
get_vector(v::AbstractVector) = v
get_vector(hv::AbstractHV) = hv.v

"""
    empty_vector(hv::AbstractHV)

Create a zero-initialized vector suitable for aggregation operations.

Returns a vector filled with the neutral element for the bundling operation 
of the given hypervector type.

# Examples
```julia
julia> empty_vector(BipolarHV(10))
10-element Vector{Int64} filled with 0s

julia> empty_vector(GradedHV(10))
10-element Vector{Float64} filled with 0.5s
```
"""
empty_vector(hv::AbstractHV) = zero(hv.v)

"""
    eldist(::Type{<:AbstractHV})
    eldist(hv::AbstractHV)

Get the element distribution used for generating random vectors of the given type.

Returns the probability distribution used to sample elements when creating new 
random hypervectors of this type.
"""
eldist(hv::AbstractHV) = eldist(typeof(hv))


# ============================================================================
# Concrete Hypervector Types
# ============================================================================

"""
    BipolarHV <: AbstractHV{Int}
    BipolarHV(n::Integer=10_000)
    BipolarHV(v::AbstractVector)
    BipolarHV(v::BitVector)

Bipolar hyperdimensional vector with elements in {-1, +1}.

Internally stored as a `BitVector` for memory efficiency, but presents elements as 
{-1, +1} through specialized indexing.

# Operations
- **Bundling**: Majority vote with tie-breaking
- **Binding**: Element-wise multiplication (equivalent to XOR on the underlying bits)
- **Similarity**: Cosine similarity

# Arguments
- `n::Integer=10_000`: Vector dimension
- `v::AbstractVector`: Convert from another vector (v > 0 → +1, v ≤ 0 → -1)
- `v::BitVector`: Use existing BitVector directly

# See also
[`BinaryHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
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

"""
    TernaryHV <: AbstractHV{Int}
    TernaryHV(n::Int=10_000)

Ternary hyperdimensional vector with elements in {-1, 0, +1}.

Currently samples only from {-1, +1} but supports zero values through operations.

# Operations
- **Bundling**: Element-wise sum with optional clamping
- **Binding**: Element-wise multiplication
- **Similarity**: Cosine similarity

# Arguments
- `n::Int=10_000`: Vector dimension

# See also
[`BipolarHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
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


"""
    BinaryHV <: AbstractHV{Bool}
    BinaryHV(n::Integer=10_000)
    BinaryHV(v::AbstractVector{Bool})

Binary hyperdimensional vector with elements in {0, 1} (false, true).

Stored as a `BitVector` for memory efficiency.

# Operations
- **Bundling**: Majority vote with tie-breaking
- **Binding**: Element-wise XOR
- **Similarity**: Jaccard similarity

# Arguments
- `n::Integer=10_000`: Vector dimension
- `v::AbstractVector{Bool}`: Use existing boolean vector

# See also
[`BipolarHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
struct BinaryHV <: AbstractHV{Bool}
    v::BitVector
end

BinaryHV(n::Integer=10_000) = BinaryHV(bitrand(n))
BinaryHV(v::AbstractVector{Bool}) = BinaryHV(BitVector(v))

# needed for aggregation
empty_vector(hv::BinaryHV) = zeros(Int, length(hv))

eldist(::Type{BinaryHV}) = Bernoulli(0.5)

"""
    RealHV{T<:Real} <: AbstractHV{T}
    RealHV(n::Integer=10_000, distr::Distribution=Normal())
    RealHV(T::Type{<:Real}, n::Integer=10_000, distr::Distribution=Normal())

Real-valued hyperdimensional vector with elements drawn from a continuous distribution.

Elements are typically drawn from a standard normal distribution, providing the richest 
representational capacity among all hypervector types.

# Operations
- **Bundling**: Element-wise sum with normalization to preserve vector norm
- **Binding**: Element-wise multiplication
- **Similarity**: Cosine similarity

# Arguments
- `n::Integer=10_000`: Vector dimension
- `T::Type{<:Real}`: Element type (default: Float64)
- `distr::Distribution`: Distribution to sample from (default: Normal())

# See also
[`GradedHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
struct RealHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
end

RealHV(n::Integer=10_000, distr::Distribution=eldist(RealHV)) = RealHV(rand(distr, n))

RealHV(T::Type{<:Real}, n::Integer=10_000, distr::Distribution=eldist(RealHV)) = RealHV(T.(rand(distr, n)))


Base.similar(hv::RealHV) = RealHV(length(hv), eldist(RealHV))

function LinearAlgebra.normalize!(hv::RealHV)
    target_std = std(eldist(RealHV))
    current_std = std(hv.v)
    if current_std > 0  # Avoid division by zero
        hv.v .*= target_std / current_std
    end
    return hv
end

eldist(::Type{<:RealHV}) = Normal()


"""
    GradedHV{T<:Real} <: AbstractHV{T}
    GradedHV(n::Int=10_000, distr=Beta(1,1))

Graded hyperdimensional vector with elements in [0, 1].

Allows for soft, graded relationships rather than hard binary associations. 
Uses specialized "3π" operation for bundling and fuzzy XOR for binding.

# Operations
- **Bundling**: 3π operation (probabilistic bundling)
- **Binding**: Fuzzy XOR
- **Similarity**: Jaccard similarity

# Arguments
- `n::Int=10_000`: Vector dimension  
- `distr`: Distribution with support in [0,1] (default: uniform via Beta(1,1))

# See also
[`GradedBipolarHV`](@ref), [`RealHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
struct GradedHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
    #GradedHV(v::AbstractVector{T}) where {T<:Real} = new{T}(clamp!(v,0,1))
end

function GradedHV(n::Int=10_000, distr=eldist(GradedHV))
    @assert 0 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [0,1]"
    return GradedHV(rand(distr, n))
end

Base.similar(hv::GradedHV) = GradedHV(length(hv), eldist(GradedHV))

# distribution used for sampling graded HVs
eldist(::Type{<:GradedHV}) = Beta(1, 1)

# neutral element of a graded HV is 0.5
empty_vector(hv::GradedHV) = fill!(zero(hv.v), 0.5)

LinearAlgebra.normalize!(hv::GradedHV) = clamp!(hv.v, 0, 1)

function Base.zeros(hv::GradedHV)
    v = similar(hv.v)
    return fill!(v, one(eltype(v)) / 2)
end

"""
    GradedBipolarHV{T<:Real} <: AbstractHV{T}
    GradedBipolarHV(n::Int=10_000, distr::Distribution=...)

Graded bipolar hyperdimensional vector with elements in [-1, 1].

Similar to `GradedHV` but with bipolar range, allowing for both positive and 
negative graded relationships.

# Operations
- **Bundling**: 3π operation adapted for bipolar range
- **Binding**: Fuzzy XOR adapted for bipolar range  
- **Similarity**: Cosine similarity

# Arguments
- `n::Int=10_000`: Vector dimension
- `distr`: Distribution with support in [-1,1]

# See also
[`GradedHV`](@ref), [`RealHV`](@ref), [`bundle`](@ref), [`bind`](@ref)
"""
struct GradedBipolarHV{T<:Real} <: AbstractHV{T}
    v::Vector{T}
    #GradedBipolarHV(v::AbstractVector{T}) where {T<:Real} = new{T}(clamp!(v,-1,1))
end

function GradedBipolarHV(n::Int=10_000, distr::Distribution=eldist(GradedBipolarHV))
    @assert -1 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [-1,1]"
    return GradedBipolarHV(rand(distr, n))
end

# distribution used for sampling graded bipolar HVs
eldist(::Type{<:GradedBipolarHV}) = 2eldist(GradedHV) - 1

#GradedBipolarHV(n::Int) = GradedBipolarHV(graded_bipol_distr, n)

Base.similar(hv::GradedBipolarHV) = GradedBipolarHV(length(hv))
LinearAlgebra.normalize!(hv::GradedBipolarHV) = clamp!(hv.v, -1, 1)



# ============================================================================
# Type Traits for Dispatch
# ============================================================================

"""
    HVTraits

Abstract type for hypervector storage traits.

Used for dispatch to select appropriate algorithms based on underlying storage.
"""
abstract type HVTraits end

"""
    HVByteVec <: HVTraits

Trait for hypervectors stored as regular byte-based vectors.

Used for most hypervector types that store elements as numeric values.
"""
struct HVByteVec <: HVTraits end

"""
    HVBitVec <: HVTraits

Trait for hypervectors stored as bit vectors.

Used for memory-efficient binary and bipolar hypervectors.
"""
struct HVBitVec <: HVTraits end

"""
    vectype(hv::AbstractHV) -> HVTraits

Get the storage trait for a hypervector type.

Returns either `HVByteVec` or `HVBitVec` depending on underlying storage.

# Examples
```julia
julia> vectype(BinaryHV(10))
HVBitVec()

julia> vectype(RealHV(10))
HVByteVec()
```
"""
vectype(::AbstractHV) = HVByteVec
vectype(::BinaryHV) = HVBitVec
vectype(::BipolarHV) = HVBitVec
