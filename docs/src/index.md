```@meta
CurrentModule = HyperdimensionalComputing
```

# HyperdimensionalComputing.jl

Hyperdimensional computing (HDC), also known as vector symbolic architectures (VSA), is a
brain-inspired paradigm that represents information as very high-dimensional vectors --
*hypervectors*, typically 10,000 dimensions. In such spaces two random vectors are almost always
nearly orthogonal, and that single fact makes it possible to superpose, associate and sequence
concepts inside one fixed-size vector without them interfering.

The result is a computing style that is fast, robust to noise, remarkably data-efficient, and
simple enough to implement in a few hundred lines -- an appealing alternative to deep learning
for structured and biological data.

## Installation

```julia
using Pkg; Pkg.add(url = "https://github.com/Kermit-UGent/HyperdimensionalComputing.jl")
```

## A first taste

```@example index
using HyperdimensionalComputing

## every object gets its own deterministic, quasi-orthogonal hypervector
cat = encode(BipolarHV, "cat")
dog = encode(BipolarHV, "dog")
similarity(cat, dog)
```

```@example index
## bundling superposes: the result resembles each of its parts
pets = bundle([cat, dog])
similarity(pets, cat), similarity(pets, dog)
```

```@example index
## binding associates, and undoes itself
role = encode(BipolarHV, :pet)
(role * cat) / role == cat
```

## How the package is organised

The package is built in three layers, and it helps to know which one you are working in:

| Layer | Signature | Members |
|:------|:----------|:--------|
| **Primitives** | hypervector → hypervector | [`bundle`](@ref) (`+`), [`bind`](@ref) (`*`), [`unbind`](@ref) (`/`), [`shift`](@ref) (`ρ`), [`perturbate`](@ref) |
| **Combinators** | collection of hypervectors → hypervector | [`multiset`](@ref), [`multibind`](@ref), [`bundlesequence`](@ref), [`bindsequence`](@ref), [`hashtable`](@ref), [`crossproduct`](@ref), [`ngrams`](@ref), [`graph`](@ref) |
| **Encoders** | **raw data** → hypervector | [`encode`](@ref) with [`KMer`](@ref)/[`NGram`](@ref)/[`Sequence`](@ref)/[`BagOfSymbols`](@ref), and the stateful [`LevelEncoder`](@ref) and [`RandomProjection`](@ref) |

Seven vector symbolic architectures are available -- [`BinaryHV`](@ref), [`BipolarHV`](@ref),
[`TernaryHV`](@ref), [`RealHV`](@ref), [`GradedHV`](@ref), [`GradedBipolarHV`](@ref) and
[`FHRR`](@ref) -- all sharing the [`AbstractHV`](@ref) interface, so an application can usually
switch between them by changing a single name.

## Where to go next

- **[Introduction to HDC](examples/introduction-to-hdc.md)** — the operations, taught by cooking
  a taco and a hamburger. Start here.
- **[Encoding data](examples/encoding-data.md)** — turning tokens, sequences, numbers and feature
  vectors into hypervectors; includes recognising languages from character k-mers.
- **[Colours](examples/colours.md)** — random projections for continuous data, associative
  memories, and the link to kernel methods.
- **[What's the Dollar of Mexico?](examples/whats-the-dollar-of-mexico.md)** — Kanerva's classic
  analogical-reasoning example.
- **[Iris dataset](examples/iris.md)** — a complete classification workflow on numeric data.
- **[API reference](api.md)** — every exported function.

## Citing

If you use this package in research, please cite the review it accompanies:

> Stock, M., Van Criekinge, W., Boeckaerts, D., Taelman, S., Van Haeverbeke, M., Dewulf, P.,
> De Baets, B. (2024). Hyperdimensional computing: A fast, robust, and interpretable paradigm for
> biological data. *PLOS Computational Biology* 20(9), e1012426.
> [doi:10.1371/journal.pcbi.1012426](https://doi.org/10.1371/journal.pcbi.1012426)
