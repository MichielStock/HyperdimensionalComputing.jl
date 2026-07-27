```@meta
CurrentModule = HyperdimensionalComputing
```

# API Reference

This page contains the complete API reference for HyperdimensionalComputing.jl.

## Types

```@docs
AbstractHV
BinaryHV
BipolarHV
TernaryHV
GradedHV
GradedBipolarHV
RealHV
FHRR
```

## Operations

```@docs
bundle
bind
unbind
shift!
shift
ρ
ρ!
normalize
normalize!
perturbate
perturbate!
```

## Inference

```@docs
similarity
δ
nearest_neighbor
```

## Combinators

Combinators take hypervectors and return a hypervector: they compose the primitive
operations into structured representations.

```@docs
multiset
multibind
bundlesequence
bindsequence
hashtable
crossproduct
ngrams
graph
```

## Encoders

Encoders take *raw data* and return a hypervector. [`encode`](@ref) is the canonical
entry point: without a strategy it is the deterministic token path, and with an
[`AbstractEncoding`](@ref) strategy it composes the token path with the combinators
above.

```@docs
encode
decode
AbstractEncoding
BagOfSymbols
Sequence
NGram
KMer
```

Stateful encoders hold hypervector state that is built once at construction — a level
set, a projection matrix — so that separately encoded values remain mutually
comparable. They are passed as the first argument of `encode`, and support the
inverse map `decode`.

```@docs
AbstractEncoder
LevelEncoder
RandomProjection
rethreshold
```

## Package extensions

`HyperdimensionalComputing.jl` has a couple of package extensions to interact with commonly used
Julia packages:

### [UnicodePlots.jl](https://juliaplots.org/UnicodePlots.jl/stable/)

```@docs
unicodeheatmap
unicodehistogram
```
