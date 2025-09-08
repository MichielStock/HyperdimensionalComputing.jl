# Hyperdimensional Computing in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://michielstock.github.io/HyperdimensionalComputing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://michielstock.github.io/HyperdimensionalComputing.jl/dev)
[![Build Status](https://github.com/dimiboeckaerts/HyperdimensionalComputing.jl/workflows/CI/badge.svg)](https://github.com/dimiboeckaerts/HyperdimensionalComputing.jl/actions)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)


This package implements special types of vectors and associated methods for hyperdimensional computing. Hyperdimensional computing (HDC) is a paragdigm to represent patterns by means of a high-dimensional vectors (typically 10,000 dimensions). Specific operations can be used to create new vectors by combining the information or encoding some kind of position. HDC is an alternative machine learning method that is extremely computationally efficient. It is inspired by the distributed, holographic representation of patterns in the brain. Typically, the high-dimensionality is more important than the nature of the operations. This package provides various types of vectors (binary, graded, bipolar...) with sensible operations for *aggragating*, *binding* and *permutation*. 

We provide a set of types of hypervectors (HVs), with the associated operations.

## Basic use

Several types of vectors are implemented. Random vectors can be initialized of different sizes.

```julia
using HyperdimensionalComputing

x = BipolarHV()  # default length is 10,000

y = BinaryHV(20)  # different length

z = RealHV(Float32)  # specify data type
```

The basic operations are `bundle` (creating a vector that is similar to the provided vectors), `bind` (creating a vector that is dissimilar to the vectors) and `circshift` (shifting the vector inplace to create a new vector). For `bundle` and `bind`, we overload `+` and `*` as binary operators, while `ρ` is an alias for `shift`.

```julia
x, y, z = GradedHV(10), GradedHV(10), GradedHV(10)

# aggregation

bundle([x, y, z])

x + y  # binary operator for bundling

# binding

bind(x, y)

x * y  # binary operator for binding

# permutation

shift(x, 2)  # circular shifts the coordinates
ρ(x, 2)  # same

ρ!(y, 2)  # inplace
```

See the table below for which operations are used for which type.

## Embedding sequences

TODO: update!

HDC is particularly powerful for embedding sequences. This is done by creating embeddings for n-grams and aggregating the n-grams found in the sequence.

```julia
# create dictionary for embedding
alphabet = "ATCG"
basis = Dict(c=>RealHDV(100) for (c, hdv) in zip(alphabet, hdvs))

sequence = "TAGTTTGAGGATCCGCTCGCTGCAACGCG"

seq_embedding = sequence_embedding(sequence, basis, 3)  # embedding using 3-grams
```
If the size of the number of n-grams is not too large, it makes sense to precompute these to speed up the encoding process.

```julia
threegrams = compute_3_grams(basis)

sequence_embedding(sequence, threegrams) 
```

## Overview of operations

| Vector | element domain | aggregate | binding | similarity |
| ------ | --------------| ---------| ----------| --------|
| `BinaryHV` | {0, 1} | majority | xor | Jaccard |
| `BipolarHV` | {-1, 1} | sum and threshold | multiply | cosine |
| `TernaryHV` | {-1, 0, 1} | sum and threshold | multiply | cosine |
| `GradedHV` | [0, 1] |  3π  | fuzzy xor | Jaccard |
| `GradedBipolarHV` | [-1, 1] | 3π  | fuzzy xor  | cosine |
| `RealHV` | real (normally distributed) | sum weighted to keep vector norm | multiply | cosine |
