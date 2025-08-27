"""
    multiset(vs::AbstractVector{<:T})::T where {T <: AbstractHDV}

Multiset of input hypervectors, bundles all the input hypervectors together.

# Arguments
- `vs::AbstractVector{<:AbstractHDV}`: Hypervectors

# Example
```
julia> vs = [BinaryHDV(10) for _ in 1:10]
10-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
 [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
 [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
 [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
 [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
 [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
 [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]

julia> multiset(vs)
10-element BinaryHDV:
 0
 1
 0
 0
 1
 0
 0
 0
 1
 0
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\oplus_{i=1}^{m} V_i
```

where `V` is the hypervector collection, `m` is the size of the hypervector collection,
`i` is the position of the entry in the collection, and `\\oplus` is the bundling operation.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.multiset.html)

# See also

- [`multibind`](@ref): Multibind encoding, binding-variant of this encoder
"""
function multiset(vs::AbstractVector{<:T})::T where {T <: AbstractHDV}
    return aggregate(vs)
end

"""
    multibind(vs::AbstractVector{<:AbstractHDV})

Binding of multiple hypervectors, binds all the input hypervectors together.

# Arguments
- `vs::AbstractVector{<:AbstractHDV}`: Hypervectors

# Examples
```
julia> vs = [BinaryHDV(10) for _ in 1:10]
10-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
 [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
 [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
 [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
 [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
 [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
 [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]

julia> multibind(vs)
10-element BinaryHDV:
 1
 1
 1
 0
 1
 1
 1
 0
 1
 0
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\otimes_{i=1}^{m} V_i
```

where `V` is the hypervector collection, `m` is the size of the hypervector collection,
`i` is the position of the entry in the collection, and `\\otimes` is the binding operation.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.multibind.html)

# See also

- [`multiset`](@ref): Multiset encoding, bundling-variant of this encoder
"""
function multibind(vs::AbstractVector{<:AbstractHDV})
    return bind(vs)
end


"""
    bundlesequence(vs::AbstractVector{<:AbstractHDV})

Bundling-based sequence. The first value is not permuted, the last value is permuted n-1 times.

# Arguments
- `vs::AbstractVector{<:AbstractHDV}`: Hypervector sequence

# Examples
```
julia> vs = [BinaryHDV(10) for _ in 1:10]
10-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
 [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
 [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
 [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
 [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
 [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
 [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]

julia> bundlesequence(vs)
10-element BinaryHDV:
 0
 1
 0
 0
 0
 0
 0
 0
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\oplus_{i=1}^{m} \\Pi(V_i, i-1)
```

where `V` is the hypervector collection, `m` is the size of the hypervector collection,
`i` is the position of the entry in the collection, and `\\oplus` and `\\Pi` are the
bundling and permutation operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.bundle_sequence.html)

# See also

- [`bindsequence`](@ref): Binding-sequence encoding, binding-variant of this encoder
"""
function bundlesequence(vs::AbstractVector{<:AbstractHDV})
    @assert length(vs) > 1 "Can't bundle sequence of a single hypervector"
    return aggregate([Π(hv, i - 1) for (i, hv) in enumerate(vs)])
end

"""
    bindsequence(vs::AbstractVector{<:AbstractHDV})

Binding-based sequence. The first value is not permuted, the last value is permuted n-1 times.

# Arguments
- `vs::AbstractVector{<:AbstractHDV}`: Hypervector sequence

# Examples
```
julia> vs = [BinaryHDV(10) for _ in 1:10]
10-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
 [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
 [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
 [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
 [0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
 [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
 [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
 [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]

julia> bindsequence(vs)
10-element BinaryHDV:
 0
 1
 1
 0
 1
 1
 0
 1
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\otimes_{i=1}^{m} \\Pi(V_i, i-1)
```

where `V` is the hypervector collection, `m` is the size of the hypervector collection,
`i` is the position of the entry in the collection, and `\\otimes` and `\\Pi` are the
binding and permutation operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.bind_sequence.html)

# See also

- [`bundlesequence`](@ref): Bundle-sequence encoding, bundling-variant of this encoder
"""
function bindsequence(vs::AbstractVector{<:AbstractHDV})
    @assert length(vs) > 1 "Can't bind sequence of a single hypervector"
    return bind([Π(hv, i - 1) for (i, hv) in enumerate(vs)])
end

"""
    hashtable(keys::T, values::T) where {T <: AbstractVector{<:AbstractHDV}}

Hash table from keys-values hypervector pairs. Keys and values must be the same length in order
to encode as hypervector.

# Arguments
- `keys::AbstractVector{<:AbstractHDV}`: Keys hypervectors
- `values::AbstractVector{<:AbstractHDV}`: Values hypervectors

# Example
```
julia> ks = [BinaryHDV(10) for _ in 1:5]
5-element Vector{BinaryHDV}:
 [0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
 [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
 [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
 [1, 0, 0, 0, 0, 1, 1, 0, 1, 0]
 [0, 1, 1, 1, 1, 0, 0, 1, 1, 1]

julia> vs = [BinaryHDV(10) for _ in 1:5]
5-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
 [1, 0, 1, 1, 1, 0, 1, 0, 0, 0]
 [0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
 [0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
 [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]

julia> hashtable(ks, vs)
10-element BinaryHDV:
 0
 0
 0
 1
 1
 0
 1
 0
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\otimes_{i=1}^{m} K_i \\otimes V_i
```

where `K` and `V` are the key and value hypervector collections, `m` is the size of the
hypervector collection, `i` is the position of the entry in the collection, and `\\otimes`
and `\\oplus` are the binding and bundling operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.hash_table.html)
"""
function hashtable(keys::T, values::T) where {T <: AbstractVector{<:AbstractHDV}}
    @assert length(keys) == length(values) "Number of keys and values aren't equal"
    return aggregate(map(bind, zip(keys, values)))
end

"""
    crossproduct(U::T, V::T) where {T <: AbstractVector{<:AbstractHDV}}

Cross product between two sets of hypervectors.


# Arguments
- `U::AbstractVector{<:AbstractHDV}`: Hypervectors
- `V::AbstractVector{<:AbstractHDV}`: Hypervectors

# Examples
```
julia> us = [BinaryHDV(10) for _ in 1:5]
5-element Vector{BinaryHDV}:
 [1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
 [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
 [0, 1, 1, 1, 1, 0, 0, 1, 1, 1]
 [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
 [1, 0, 0, 1, 0, 0, 1, 1, 1, 1]

julia> vs = [BinaryHDV(10) for _ in 1:5]
5-element Vector{BinaryHDV}:
 [0, 1, 1, 1, 1, 1, 0, 1, 0, 0]
 [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
 [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
 [1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
 [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]

julia> crossproduct(us, vs)
10-element BinaryHDV:
 0
 0
 1
 0
 1
 0
 0
 1
 0
 1
```

# Extended help

This encoding strategy first creates a multiset from both input hypervector sets,
which are then bound together to generate all cross products, i.e.

U₁ × V₁ + U₁ × V₂ + ... + U₁ × Vₘ + ... + Uₙ × Vₘ

This encoding is based on the following formula:

```math
(\\oplus_{i=1}^{m} U_i) \\otimes (\\oplus_{i=1}^{n} V_i)
```

where U and V are collections of hypervectors, `m` and `n` are the sizes of the U and V collections,
`ì` is the position in the hypervector collection, and `\\oplus` and `\\otimes` are the bundling
and binding operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.cross_product.html)
"""
function crossproduct(U::T, V::T) where {T <: AbstractVector{<:AbstractHDV}}
    # TODO: This should be bundled without normalizing
    return aggregate([multiset(U), multiset(V)]; norm = false)
end

"""
    ngrams(vs::AbstractVector{<:AbstractHDV}, n::Int = 3)

Creates a hypervector with the _n_-gram statistics of the input.

# Arguments
- `vs::AbstractVector{<:AbstractHDV}`: Hypervector collection
- `n::Int = 3`: _n_-gram size

# Examples
```
julia> vs = [BinaryHDV(10) for _ in 1:10]
10-element Vector{BinaryHDV}:
 [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
 [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
 [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
 [1, 0, 0, 0, 1, 0, 0, 1, 1, 1]
 [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
 [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
 [1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
 [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
 [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
 [1, 1, 0, 0, 0, 1, 1, 1, 0, 1]

julia> ngrams(vs)
10-element BinaryHDV:
 1
 1
 1
 1
 1
 1
 0
 1
 0
 1
```

# Extended help

This encoding is defined by the following mathematical notation:

```math
\\oplus_{i=1}^{m-n}\\otimes_{j=1}^{n-1}\\Pi^{n-j-1}(V_{i+j})
```

where `V` is the collection of hypervectors, `m` is the number of hypervectors in the
collection `V`, `n` is the window size, `i` is the position in the sequence, `j` is the
position in the _n_-gram, and `\\oplus`, `\\otimes` and `\\Pi` are the bundling, binding
and permutation operations.

!!! note
    - For `n = 1` use `multiset()` instead
    - For `n = m` use `bindsequence()` instead

# See also

- [`multiset`](@ref): Multiset encoding, equivalent to `ngram(vs, 1)`
- [`bindsequence`](@ref): Bind-sequence encoding, equivalent to `ngram(vs, length(vs))`

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.ngrams.html)

"""
function ngrams(vs::AbstractVector{<:AbstractHDV}, n::Int = 3)
    m = length(vs)
    @assert 1 <= n <= length(vs) "`n` must be 1 ≤ n ≤ $m"
    return aggregate([bind([Π(vs[i + j], j - 1) for j in 1:n]) for i in 1:(m - n)])
end

"""
    graph(source::T, target::T, directed::Bool = false)

Graph for `source`-`target` pairs. Can be directed or undirected.

# Arguments
- `source::T`: Source node hypervectors
- `target::T`: Target node hypervectors
- `directed::Bool = false`: Whether the graph is directed or not

# Example

# Extended help

This encoding is based on the following mathematical notation:

*Undirected graphs*
```math
\\otimes_{i=1}^{m} S_i \\otimes T_i
```

*Directed graphs*
```math
\\otimes_{i=1}^{m} S_i \\otimes \\Pi(T_i)
```

where `K` and `V` are the key and value hypervector collections, `m` is the size of the
hypervector collection, `i` is the position of the entry in the collection, and `\\otimes`,
`\\oplus` and `\\Pi` are the binding, bundling and permutation operations.

# See also

- [`hashtable`](@ref): Hash table encoding, underlying encoding strategy of this encoder.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.graph.html)

"""
function graph(source::T, target::T; directed::Bool = false) where {T <: AbstractVector{<:AbstractHDV}}
    @assert length(source) == length(target) "`source` and `target` must be the same length"
    return hashtable(source, Π.(target, convert(Int, directed)))
end
