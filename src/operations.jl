"""
    operations.jl

This module implements the core operations on hypervectors that enable hyperdimensional computing.
Hyperdimensional computing relies on three fundamental operations that work with high-dimensional vectors 
(typically 10,000 dimensions) representing symbols, concepts, or data.

## Core Operations

### 1. Bundling/Aggregation (`+`)
Combines information from multiple hypervectors into a new vector that is similar to all inputs.
This operation is used to create superposition states and aggregate information.
- **Symbol**: `+` (addition)
- **Properties**: Commutative, approximately associative (exact associativity depends on HV type)
- **Use case**: Combining multiple concepts or creating set representations

### 2. Binding (`*`) 
Combines two hypervectors into a new vector that is dissimilar from both inputs.
This operation creates structured representations and enables compositional encoding.
- **Symbol**: `*` (multiplication) or `bind()`
- **Properties**: Distributes over bundling, preserves distances, typically commutative
- **Use case**: Creating associations, encoding sequences, structural relationships

### 3. Shifting/Permutation (`ρ`)
Circularly shifts the elements of a hypervector, creating a new vector with the same statistical properties.
- **Symbol**: `ρ` (rho) or `shift()`
- **Properties**: Distributes over bundling, preserves distances and sparsity
- **Use case**: Positional encoding, creating systematic variations

## Hypervector Types

Different hypervector types use specialized implementations:
- **BinaryHV**: XOR for binding, majority vote for bundling
- **BipolarHV**: XOR for binding, thresholded sum for bundling  
- **TernaryHV**: Element-wise multiplication for binding, addition for bundling
- **RealHV**: Element-wise multiplication for binding, normalized addition for bundling
- **GradedHV**: Fuzzy operations for binding, probabilistic three-way operations for bundling
- **GradedBipolarHV**: Similar to GradedHV but in bipolar domain [-1,1]

## Implementation Notes

The operations are implemented to be type-stable and efficient, with specialized methods for each 
hypervector type to leverage their specific mathematical properties and storage formats.
For binary/bipolar types, tie-breaking is handled using random bits when bundling even numbers of vectors.
"""

# UTILITY FUNCTIONS FOR GRADED OPERATIONS
# =======================================

"""
    grad2bipol(x::Real)

Maps a graded number from the [0, 1] interval to the bipolar [-1, 1] interval.

This transformation is used to convert between GradedHV and GradedBipolarHV representations.
The mapping is linear: `f(x) = 2x - 1`.

# Examples
```julia
grad2bipol(0.0)   # -1.0
grad2bipol(0.5)   #  0.0  
grad2bipol(1.0)   #  1.0
```
"""
grad2bipol(x::Real) = 2x - one(x)

"""
    bipol2grad(x::Real)

Maps a bipolar number from the [-1, 1] interval to the graded [0, 1] interval.

This is the inverse transformation of `grad2bipol`. The mapping is linear: `f(x) = (x + 1)/2`.

# Examples
```julia
bipol2grad(-1.0)  # 0.0
bipol2grad(0.0)   # 0.5
bipol2grad(1.0)   # 1.0
```
"""
bipol2grad(x::Real) = (x + one(x)) / 2

"""
    three_pi(x, y)

Implements the "three π" operation for graded hypervectors, a key operation in fuzzy logic.

This operation is used for bundling GradedHV hypervectors. It behaves as:
- When |x - y| = 1 (one is 0, the other is 1): returns 0
- Otherwise: returns the geometric mean weighted by confidence

The formula is: `x * y / (x * y + (1-x) * (1-y))` when the denominator is non-zero.

# Mathematical Properties
- Commutative: three_pi(x, y) = three_pi(y, x)
- Self-inverse with complement: three_pi(x, 1-x) = 0
- Preserves extreme values: three_pi(1, 1) = 1, three_pi(0, 0) = 0
"""
three_pi(x, y) = abs(x - y) == 1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))

"""
    fuzzy_xor(x, y)

Implements fuzzy XOR operation for graded hypervectors.

This operation is used for binding GradedHV hypervectors. It computes:
`(1-x) * y + x * (1-y)`

This represents the probability that exactly one of x or y is true in fuzzy logic,
making it suitable for binding operations that should be dissimilar from inputs.

# Mathematical Properties  
- Commutative: fuzzy_xor(x, y) = fuzzy_xor(y, x)
- Self-inverse: fuzzy_xor(fuzzy_xor(x, y), y) ≈ x
- Boundary behavior: fuzzy_xor(0, y) = y, fuzzy_xor(1, y) = 1-y
"""
fuzzy_xor(x, y) = (one(x) - x) * y + x * (one(y) - y)

"""
    three_pi_bipol(x, y)

Implements three π operation in the bipolar domain [-1, 1].

This converts bipolar inputs to graded domain, applies three_pi, then converts back.
Used for bundling GradedBipolarHV hypervectors.
"""
three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))

"""
    fuzzy_xor_bipol(x, y)

Implements fuzzy XOR operation in the bipolar domain [-1, 1].

This converts bipolar inputs to graded domain, applies fuzzy_xor, then converts back.
Used for binding GradedBipolarHV hypervectors.
"""
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))

# OPERATION DISPATCH FUNCTIONS
# ============================

"""
    aggfun(hv::AbstractHV)
    aggfun(::Type{<:AbstractHV})

Returns the appropriate aggregation/bundling function for a hypervector type.

Different hypervector types use different aggregation strategies:
- **Default**: Element-wise addition (`+`)
- **GradedHV**: Three-pi operation (`three_pi`) for probabilistic bundling
- **GradedBipolarHV**: Bipolar three-pi operation (`three_pi_bipol`)

# Examples
```julia
aggfun(BinaryHV)           # Returns +
aggfun(GradedHV)           # Returns three_pi
aggfun(GradedBipolarHV)    # Returns three_pi_bipol
```
"""
aggfun(::Type{<:AbstractHV}) = +
aggfun(::GradedHV) = three_pi
aggfun(::GradedBipolarHV) = three_pi_bipol

"""
    bindfun(hv::AbstractHV)

Returns the appropriate binding function for a hypervector type.

Different hypervector types use different binding strategies:
- **Default**: Element-wise multiplication (`*`)
- **BinaryHV**: XOR operation (`⊻`)
- **GradedHV**: Fuzzy XOR (`fuzzy_xor`) for graded binding
- **GradedBipolarHV**: Bipolar fuzzy XOR (`fuzzy_xor_bipol`)

# Examples
```julia
bindfun(RealHV(1000))         # Returns *
bindfun(BinaryHV(1000))       # Returns ⊻
bindfun(GradedHV(1000))       # Returns fuzzy_xor
```
"""
bindfun(::AbstractHV) = *
bindfun(::BinaryHV) = ⊻
bindfun(::GradedHV) = fuzzy_xor
bindfun(::GradedBipolarHV) = fuzzy_xor_bipol

"""
    neutralbind(hdv::AbstractHV)

Returns the neutral element for binding operations for the given hypervector type.

The neutral element `e` satisfies: `bind(hv, e) = hv` for any hypervector `hv`.

# Neutral Elements by Type
- **Default**: `1` (multiplicative identity)
- **BinaryHV**: `false` (XOR identity) 
- **GradedHV**: `0` (fuzzy XOR identity in [0,1])
- **GradedBipolarHV**: `-1` (fuzzy XOR identity in [-1,1])

# Examples
```julia
neutralbind(RealHV(1000))      # Returns 1.0
neutralbind(BinaryHV(1000))    # Returns false
neutralbind(GradedHV(1000))    # Returns 0.0
```
"""
neutralbind(hdv::AbstractHV) = one(eltype(hdv))
neutralbind(hdv::BinaryHV) = false
neutralbind(hdv::GradedHV) = zero(eltype(hdv))
neutralbind(hdv::GradedBipolarHV) = -one(eltype(hdv))


# BUNDLING/AGGREGATION OPERATIONS  
# ================================

"""
    bundle(hdvs; kwargs...)
    bundle(prototype_hv, hdvs, accumulator; kwargs...)

Combines multiple hypervectors into a single hypervector that is similar to all inputs.

This is the core aggregation operation in hyperdimensional computing, used to create
superposition states and combine information from multiple sources.

# Type-Specific Implementations

- **BinaryHV/BipolarHV**: Uses majority voting with random tie-breaking
- **TernaryHV**: Simple addition with optional normalization
- **RealHV**: Addition normalized by √m for m vectors
- **GradedHV**: Iterative three-pi operations
- **GradedBipolarHV**: Iterative bipolar three-pi operations

# Arguments
- `hdvs`: Collection of hypervectors to bundle
- `prototype_hv`: Template hypervector determining the return type
- `accumulator`: Pre-allocated accumulator vector (for efficiency)
- `normalize=false`: Whether to normalize result (TernaryHV only)

# Examples
```julia
hv1, hv2, hv3 = BinaryHV(1000), BinaryHV(1000), BinaryHV(1000)
bundled = bundle([hv1, hv2, hv3])

# Equivalent using + operator
bundled = hv1 + hv2 + hv3
```

# Mathematical Properties
- Commutative: order of inputs doesn't affect result (up to tie-breaking)
- Similarity: result is similar to all input vectors
- Distributive: binding distributes over bundling
"""
function bundle(hdvs; kwargs...)
    hv = first(hdvs)
    r = empty_vector(hv)
    return bundle(hv, hdvs, r; kwargs...)
end

# Binary and Bipolar HV: use majority voting
function bundle(hvr::Union{BinaryHV,BipolarHV}, hdvs, r)
    m = length(hdvs)
    for hv in hdvs
        r .+= hv.v
    end
    if iseven(m)  # break ties randomly
        r .+= bitrand(length(r))
    end
    hvr = similar(hvr)
    hvr.v .= r .> m / 2
    return hvr
end

# TernaryHV: simple addition with optional normalization
function bundle(
    ::TernaryHV, hdvs, r;
    normalize=false
)
    for hv in hdvs
        r .+= hv.v
    end
    normalize && clamp!(r, -1, 1)
    return TernaryHV(r)
end

# RealHV: addition normalized by √m
function bundle(::RealHV, hdvs, r)
    m = 0
    for hv in hdvs
        r .+= hv.v
        m += 1
    end
    r ./= sqrt(m)
    return RealHV(r)
end

# GradedHV: iterative three-pi operations
function bundle(::GradedHV, hdvs, r)
    for hv in hdvs
        r .= three_pi.(r, hv.v)
    end
    return GradedHV(r)
end

# GradedBipolarHV: iterative bipolar three-pi operations  
function bundle(::GradedBipolarHV, hdvs, r)
    for hv in hdvs
        r .= three_pi_bipol.(r, hv.v)
    end
    return GradedBipolarHV(r)
end

"""
    +(hv1::AbstractHV, hv2::AbstractHV)

Bundles two hypervectors using the `+` operator.

This provides a convenient syntax for bundling operations: `hv1 + hv2` is equivalent
to `bundle([hv1, hv2])`. The operation is commutative but not generally associative
due to tie-breaking mechanisms in binary/bipolar types.
"""
Base.:+(hv1::HV, hv2::HV) where {HV<:AbstractHV} = bundle((hv1, hv2))

# BINDING OPERATIONS
# ==================

"""
    bind(hv1::AbstractHV, hv2::AbstractHV)
    *(hv1::AbstractHV, hv2::AbstractHV)

Binds two hypervectors to create a new vector dissimilar from both inputs.

Binding is the core compositional operation in hyperdimensional computing, used to
create structured representations and encode relationships between concepts.

# Type-Specific Implementations

- **BinaryHV**: XOR operation (⊻) - creates random-looking result
- **BipolarHV**: XOR operation (⊻) - leverages bit representation
- **TernaryHV**: Element-wise multiplication - preserves ternary values  
- **RealHV**: Element-wise multiplication - standard vector multiplication
- **GradedHV**: Fuzzy XOR - probabilistic binding for graded values
- **GradedBipolarHV**: Bipolar fuzzy XOR - graded binding in bipolar domain

# Mathematical Properties

- **Dissimilarity**: bind(hv1, hv2) is dissimilar from both hv1 and hv2
- **Commutativity**: bind(hv1, hv2) = bind(hv2, hv1) (for most types)
- **Distributivity**: bind(hv1, bundle([hv2, hv3])) ≈ bundle([bind(hv1, hv2), bind(hv1, hv3)])
- **Invertibility**: For many types, bind(bind(hv1, hv2), hv2) ≈ hv1
- **Distance preservation**: distances between bound vectors reflect original distances

# Examples
```julia
hv1 = BinaryHV(1000)
hv2 = BinaryHV(1000)
bound = bind(hv1, hv2)  # or hv1 * hv2

# Binding is approximately self-inverse
unbound = bind(bound, hv2)
@assert unbound ≈ hv1  # Should be similar to original hv1
```

# Use Cases
- Encoding sequences: bind(item, shift(position))
- Creating associations: bind(key, value)
- Compositional representations: bind(noun, bind(verb, object))
"""

# BinaryHV: XOR binding
Base.bind(hv1::BinaryHV, hv2::BinaryHV) = BinaryHV(hv1.v .⊻ hv2.v)

# BipolarHV: XOR binding (using underlying BitVector)
Base.bind(hv1::BipolarHV, hv2::BipolarHV) = BipolarHV(hv1.v .⊻ hv2.v)

# TernaryHV: element-wise multiplication
Base.bind(hv1::TernaryHV, hv2::TernaryHV) = TernaryHV(hv1.v .* hv2.v)

# RealHV: element-wise multiplication
Base.bind(hv1::RealHV, hv2::RealHV) = RealHV(hv1.v .* hv2.v)

# GradedHV: fuzzy XOR for graded binding
Base.bind(hv1::GradedHV, hv2::GradedHV) = GradedHV(fuzzy_xor.(hv1.v, hv2.v))

# GradedBipolarHV: bipolar fuzzy XOR
Base.bind(hv1::GradedBipolarHV, hv2::GradedBipolarHV) = GradedBipolarHV(fuzzy_xor_bipol.(hv1.v, hv2.v))

"""
    *(hv1::AbstractHV, hv2::AbstractHV)

Convenience operator for binding hypervectors.

Equivalent to `bind(hv1, hv2)` but provides familiar multiplication syntax.
Note: This is not standard scalar multiplication but hypervector binding.
"""
Base.:*(hv1::HV, hv2::HV) where {HV<:AbstractHV} = bind(hv1, hv2)


# SHIFTING/PERMUTATION OPERATIONS
# ===============================

"""
    shift(hv::AbstractHV, k=1)
    shift!(hv::AbstractHV, k=1)
    ρ(hv::AbstractHV, k=1)  
    ρ!(hv::AbstractHV, k=1)

Circularly shifts the elements of a hypervector by k positions.

Shifting (also called permutation) is the third fundamental operation in hyperdimensional
computing. It creates systematic variations of hypervectors while preserving their
statistical properties and distances.

# Mathematical Properties

- **Distance preservation**: shift(hv1, k) and shift(hv2, k) have the same distance as hv1 and hv2
- **Distributivity over bundling**: shift(bundle([hv1, hv2]), k) = bundle([shift(hv1, k), shift(hv2, k)])
- **Invertibility**: shift(shift(hv, k), -k) = hv
- **Statistical preservation**: shifted vectors have same statistical properties as originals
- **Systematic variation**: small shifts create systematically related vectors

# Arguments  
- `hv`: Input hypervector to shift
- `k`: Number of positions to shift (default: 1)
  - Positive k: shift elements to the right (towards higher indices)
  - Negative k: shift elements to the left (towards lower indices)
  - k=0: returns copy of original vector

# Variants
- `shift(hv, k)`: Returns new shifted hypervector (non-mutating)
- `shift!(hv, k)`: Modifies hv in-place (mutating)
- `ρ(hv, k)`: Greek letter rho, common notation for permutation (non-mutating)
- `ρ!(hv, k)`: In-place permutation (mutating)

# Implementation Notes
- Uses efficient circular shifting via `circshift!`
- BinaryHV and BipolarHV have specialized implementations for BitVector efficiency
- All operations preserve the hypervector type

# Examples
```julia
hv = BinaryHV([true, false, true, false])
shifted = shift(hv, 1)     # [false, true, false, true]
shifted2 = ρ(hv, -1)      # [false, true, false, true] (equivalent)

# Invertibility
original = shift(shift(hv, 3), -3)
@assert original == hv

# Distance preservation  
hv1, hv2 = BinaryHV(1000), BinaryHV(1000)
dist_orig = sum(hv1.v .!= hv2.v)
dist_shifted = sum(shift(hv1, 5).v .!= shift(hv2, 5).v)  
@assert dist_orig == dist_shifted
```

# Use Cases
- **Positional encoding**: bind(word, shift(position_hv, word_index))
- **Sequence encoding**: Creating systematic position representations
- **Temporal encoding**: Representing time-dependent relationships
- **Spatial encoding**: Encoding spatial relationships and transformations
"""

# Generic implementation for most hypervector types
function shift(hv::AbstractHV, k=1)
    r = similar(hv)
    r.v .= circshift(hv.v, k)
    return r
end

# Generic in-place implementation
shift!(hv::AbstractHV, k=1) = (circshift!(hv.v, k); hv)

# Optimized implementation for BitVector-based types (BinaryHV, BipolarHV)
function shift(hv::V, k=1) where {V<:Union{BinaryHV,BipolarHV}}
    v = similar(hv.v)  # Create empty bitvector
    return V(circshift!(v, hv.v, k))
end

# Optimized in-place implementation for BitVector-based types
function shift!(hv::V, k=1) where {V<:Union{BinaryHV,BipolarHV}}
    v = similar(hv.v)  # Create empty bitvector
    hv.v .= circshift!(v, hv.v, k)
    return hv
end

# Greek letter aliases (common in HDC literature)
"""
    ρ(hv::AbstractHV, k=1)

Greek letter rho (ρ) - common notation for permutation/shifting in HDC literature.
Equivalent to `shift(hv, k)`.
"""
ρ(hv::AbstractHV, k=1) = shift(hv, k)

"""
    ρ!(hv::AbstractHV, k=1)

In-place version of ρ (rho) permutation. Equivalent to `shift!(hv, k)`.
"""
ρ!(hv::AbstractHV, k=1) = shift!(hv, k)


# COMPARISON AND SIMILARITY
# =========================

"""
    isequal(hv1::AbstractHV, hv2::AbstractHV)
    ==(hv1::AbstractHV, hv2::AbstractHV)

Tests exact equality between two hypervectors.

Two hypervectors are considered equal if and only if all their elements are identical.
This is a strict comparison that requires perfect matches.

# Examples
```julia
hv1 = BinaryHV([true, false, true])
hv2 = BinaryHV([true, false, true])
hv3 = BinaryHV([true, false, false])

hv1 == hv2  # true
hv1 == hv3  # false
```
"""
Base.isequal(hv1::AbstractHV, hv2::AbstractHV) = hv1.v == hv2.v

"""
    isapprox(u::BinaryHV, v::BinaryHV; pval::Float64, rtol)
    isapprox(u::BipolarHV, v::BipolarHV; pval::Float64, rtol)

Statistical similarity test for binary/bipolar hypervectors using p-value threshold.

Uses a binomial test under the null hypothesis that the vectors are unrelated (random).
For random binary vectors, mismatches follow Binomial(N, 0.5).

# Parameters
- `pval`: P-value threshold for statistical significance (e.g., 0.01)
- `rtol`: tolerance for fraction of mismatches. Bypasses the statistical test when given

# Examples
```julia
hv1 = BinaryHV(10000)
hv2 = BinaryHV(10000)  # Random, unrelated
hv3 = hv1 + perturbate(hv1, 0.05)  # Similar to hv1 
normalize!(hv3)

isapprox(hv1, hv2; pval=0.01)  # false - random vectors
isapprox(hv1, hv3; pval=0.01)  # true - hv3 is similar to hv1
```
"""
function Base.isapprox(u::T, v::T; pval::Float64=0.01, rtol=nothing) where {T<:Union{BinaryHV,BipolarHV}}
    @assert length(u) == length(v) "Vectors must be of equal length"
    @assert 0 < pval < 1 "pval must be between 0 and 1"

    N = length(u)
    mismatches = sum(ui != vi for (ui, vi) in zip(u, v))

    # checks number of mismatches
    !isnothing(rtol) && return mismatches / N < rtol

    # One-tailed test: probability of seeing this many or fewer mismatches by chance
    # Under H₀: mismatches ~ Binomial(N, 0.5)
    p_observed = cdf(Binomial(N, 0.5), mismatches)
    return p_observed < pval
end


"""
    isapprox(u::AbstractHV, v::AbstractHV; threshold=0.1)

Similarity-based test for general hypervectors.

Computes similarity between vectors and compares against a threshold.
The similarity function will be defined separately for different vector types.

# Parameters
- `threshold`: Minimum similarity score for vectors to be considered similar (default: 0.1)

# Examples
```julia
hv1 = RealHV(1000)
hv2 = RealHV(1000)  # Random, unrelated
hv3 = 0.9 * hv1 + 0.1 * RealHV(1000)  # Similar to hv1

hv1 ≈ hv2  # false - low similarity
hv1 ≈ hv3  # true - high similarity  
```

# Notes
- Uses `similarity(u, v)` function which needs to be implemented
- Threshold of 0.1 is a reasonable default for most applications
- Different similarity measures may be appropriate for different vector types
"""
function Base.isapprox(u::T, v::T; threshold=0.1) where {T<:AbstractHV}
    @assert length(u) == length(v) "Vectors must be of equal length"
    @assert 0 <= threshold <= 1 "threshold must be between 0 and 1"

    sim = similarity(u, v)
    return sim > threshold
end


# PERTURBATION AND NOISE
# ======================

"""
    randbv(n::Int, m::Int)
    randbv(n::Int, p::Number)
    randbv(n::Int, indices)

Generates random binary masks for perturbation operations.

Creates BitVectors that can be used to specify which elements of a hypervector
should be modified during perturbation.

# Methods
- `randbv(n, m)`: Create mask with exactly `m` true elements out of `n` total
- `randbv(n, p)`: Create mask with approximately `p*n` true elements (probability `p`)
- `randbv(n, indices)`: Create mask with true elements at specified indices

# Arguments
- `n`: Length of the binary vector
- `m`: Exact number of true elements
- `p`: Probability of each element being true (must be in [0,1])
- `indices`: Collection of indices to set to true

# Examples
```julia
mask1 = randbv(1000, 50)    # Exactly 50 true elements
mask2 = randbv(1000, 0.05)  # Approximately 5% true elements  
mask3 = randbv(1000, [1, 5, 10])  # True at indices 1, 5, 10
```
"""
function randbv(n::Int, m::Int)
    v = falses(n)  # Create empty bitvector
    v[1:m] .= true  # Set first m elements to true
    return shuffle!(v)  # Randomize positions
end

function randbv(n::Int, p::Number)
    @assert 0 ≤ p ≤ 1 "p should be a valid probability in [0,1]"
    return randbv(n, round(Int, p * n))
end

function randbv(n::Int, indices)
    v = falses(n)
    v[indices] .= true
    return v
end

"""
    perturbate!(hv::AbstractHV, mask, [dist])
    perturbate!(hv::AbstractHV, probability, [dist]) 
    perturbate!(hv::AbstractHV, indices, [dist])

In-place perturbation of hypervector elements.

Modifies selected elements of a hypervector according to the specified mask or pattern.
Different hypervector types use different perturbation strategies based on their 
underlying representation.

# Perturbation Strategies by Type

## Byte-based vectors (RealHV, TernaryHV, GradedHV, GradedBipolarHV)
- Replace masked elements with new random values from the type's distribution
- Uses `eldist(hv)` to determine appropriate random distribution
- Maintains type-specific constraints (e.g., [-1,1] for TernaryHV)

## Bit-based vectors (BinaryHV, BipolarHV) 
- Flip (XOR) masked elements with random bits
- Preserves the bit-vector structure and efficiency
- Uses XOR operation to toggle selected bits

# Arguments
- `hv`: Hypervector to modify (modified in-place)
- `mask`: BitVector indicating which elements to perturbate
- `probability`: Float in [0,1] - fraction of elements to perturbate
- `indices`: Collection of specific indices to perturbate  
- `dist`: Optional distribution for sampling new values (byte-based types only)

# Examples
```julia
hv = RealHV(1000)
original = copy(hv)

# Perturbate 5% of elements
perturbate!(hv, 0.05)

# Perturbate specific indices
perturbate!(hv, [1, 10, 100])

# Perturbate using custom mask
mask = randbv(1000, 50)
perturbate!(hv, mask)

# For binary vectors, perturbation flips bits
bhv = BinaryHV(1000) 
perturbate!(bhv, 0.1)  # Flip ~10% of bits
```

# Notes
- Modifies the hypervector in-place for efficiency
- Use `perturbate()` (without !) for non-mutating version
- Perturbation strength affects similarity to original vector
- Small perturbations preserve most properties, large ones create dissimilar vectors
"""

# Byte-based vectors: replace elements with random values
function perturbate!(::Type{HVByteVec}, hv::HV, indices, dist=eldist(hv)) where {HV<:AbstractHV}
    hv.v[indices] .= rand(dist, length(indices))
    return hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, mask::BitVector, dist=eldist(hv)) where {HV<:AbstractHV}
    hv.v[mask] .= rand(dist, sum(mask))
    return hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, p::Number, args...) where {HV<:AbstractHV}
    return perturbate!(hv, randbv(length(hv), p), args...)
end

# Bit-based vectors: XOR with random mask
function perturbate!(::Type{HVBitVec}, hv::AbstractHV, mask_spec)
    n = length(hv)
    mask = randbv(n, mask_spec)  # Convert specification to binary mask
    hv.v .⊻= mask  # XOR with mask to flip selected bits
    return hv
end

# Dispatch to appropriate implementation based on vector type
perturbate!(hv, args...) = perturbate!(vectype(hv), hv, args...)

"""
    perturbate(hv::AbstractHV, args...; kwargs...)

Non-mutating version of perturbation.

Creates a copy of the hypervector and applies perturbation to the copy,
leaving the original vector unchanged.

# Examples
```julia
hv = BinaryHV(1000)
perturbed = perturbate(hv, 0.1)  # hv remains unchanged

@assert hv != perturbed  # Different vectors
@assert hv ≈ perturbed   # But statistically similar
```
"""
perturbate(hv::AbstractHV, args...; kwargs...) = perturbate!(copy(hv), args...; kwargs...)
