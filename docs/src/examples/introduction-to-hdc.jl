using Handcalcs #hide

# # Introduction
#
# Hyperdimensional Computing (HDC) is a brain-inspired computational paradigm that represents
# and manipulates information using high-dimensional vectors called **hypervectors**. These
# vectors typically have thousands of dimensions (often 1,000-10,000), making them
# "hyperdimensional."
#
# The key insight is that high-dimensional spaces have unique mathematical properties that allow
# for robust, fault-tolerant computation.
#
# Let's start by loading the package in question, as follows:

using HyperdimensionalComputing

ρ(hv::AbstractHDV, n::Int = 1) = Π(hv, n) #hide

# # Creating hypervectors
#
# First, we will create a random bipolar hypervector. This is done as follows:

BipolarHDV()

# As you may see, by default the hypervector created has 10.000 dimensions. This is the default
# value in `HyperdimensionalComputing.jl`, but one can can create a hypervector of any given
# dimensionality by providing the size of this as an argument:

BipolarHDV(8)

# Alternatively, one can create a hypervector directly from a `AbstractVector`:

BipolarHDV(rand([-1, 1], 8))

# Let's create 3 bipolar hypervector to use for the tutorial:

h₁ = BipolarHDV(8)
h₂ = BipolarHDV(8)
h₃ = BipolarHDV(8);


# The package has different hypervector types, such as `BipolarHDV`, `TernaryHDV`, `RealHDV`,
# `GradedBipolarHDV`, and `GradedHDV`. All of this hypervectors have a common abstract type
# `AbstractHDV` which can be used to build additional functions or encoding strategies (more on
# both later).
#
# !!! info "On (abstract) types"
#     All hypervectors implemented on `HyperdimensionalComputing.jl` can be found by checking the
#     docstrings for the `AbstractHDV` (by typing `?AbstractHDV` on the Julia REPL).
#
#     For more information on a specific hypervector type, the docstrings contain information on
#     the implementation, operations, similarity measurement and other technical
#     characteristics.
#
# # Fundamental operations with hypervectors
#
# HDC uses three primary operations that preserve the hyperdimensional properties and allow for
# the representation more complex structures:
#
# ## Bundling
#
# Bundling (also known as superposition) combines multiple hypervectors to create a new hypervector
# that is similar to it's constituyents.
#
# $$u = [h_1 + h_2 + h_3]$$
#
# where $[...]$ denotes a potential normalization operations. In the case of bipolar
# hypervectors, this normalization operation is the `sign` function, which is defined as
# follows:
#
# $$\text{sign}(i) = \begin{cases}
#   +1 & \text{if } i > 0 \\
#   -1 & \text{if } i < 0 \\
#    0 & \text{otherwise }
# \end{cases}$$
#
# In HyperdimensionalComputing.jl, you can bundle hypervectors as follows:

aggregate([h₁, h₂, h₃])

# alternatively, you can use the `+` operator (which if overloaded for all `AbstractHDV`):

h₁ + h₂ + h₃

# This operation generates a hypervector that is similar to all it's contituyent hypervectors,
# such that
#
# $$h₁ \sim u, h₂ \sim u, h₃ \sim u$$
#
# where $\sim$ means that the hypervectors are similar, i.e. they share more components than
# expected by chance.
#
# ## Binding
#
# Binding combines multiple hypervectors to create a new hypervector that is dissimilar to it's
# constituyents, such that:
#
# $$v = [h₁ \times h₂ \times h₃]$$
#
# where $[...]$ represents a normalization procedure.
#
# In HyperdimensionalComputing.jl, you can bind hypervectors as follows:

bind([h₁, h₂, h₃])

# alternatively, you can use the `*` operator (which if overloaded for all `AbstractHDV`):

h₁ * h₂ * h₃

# This operation generates a hypervector that is similar to all it's contituyent hypervectors,
# such that
#
# $$h₁ \nsim v, h₂ \nsim v, h₃ \nsim v$$
#
# where $\nsim$ means that the hypervectors are dissimilar, i.e. they are quasi-orthogonal.
#
# ## Permutation
#
# Permutation (also known as shifting) is a special case of binding that creates a variant of a
# single hypervector via, generally speaking, a circular vector shifting with one or more
# positions.
#
# $$m = \rho(h₁)$$

@handcalcs h₁ # hide

#

@handcalcs ρ(h₁) # hide

#

@handcalcs h₁ != ρ(h₁) # hide

#
# The new hypervector will be, in principle, dissimilar to it's original version, such that:
#
# $$h_1 \nsim \rho(h_1) \nsim \rho\rho(h_1) \nsim \rho\rho(h_1) ...$$
#
# where $\nsim$ means that the hypervectors are dissimilar, i.e. they are quasi-orthogonal.
#
# In `HyperdimensionalComputing.jl`, one can shift hypervector as follows:

ρ(h₁, 1)

#

h₁ != ρ(h₁, 1) != ρ(h₁, 2) != ρ(h₁, 3)

# ## Similarity
#
# Althought technically not an operation, in order to retrieve information from hypervectors,
# we need to compare them using similarity/distance functions. `HyperdimensionalComputing.jl`
# provides a handy `similarity` function that accepts:
#
# 2 hypervectors:

similarity(h₁, h₂)

# A vector of hypervectors:

similarity(h₁, h₁)

# or a hypervector and a vector of hypervectors:

similarity.(Ref(h₁), [h₁, h₂, h₃])


# ## Encoding things as hypervectors
#
# The true power of HDC emerges when we combine the fundamental operations to encode complex data
# structures as hypervectors. By creatively applying bundling, binding, and shifting, we can
# represent virtually any type of information - from sequences and hierarchies to graphs and
# associative memories. The operations act as building blocks that can be composed in countless
# ways, limited only by our imagination and the specific requirements of our application. Let's
# explore some fundamental encoding strategies that demonstrate this flexibility.
#
# ### Key-value pairs
#
# Animal hypervectors:

dog_hv = BipolarHDV()
cat_hv = BipolarHDV()
cow_hv = BipolarHDV()
animals = [dog_hv, cat_hv, cow_hv]

# Sound hypervectors:

bark_hv = BipolarHDV()
meow_hv = BipolarHDV()
moo_hv = BipolarHDV()
sounds = [bark_hv, meow_hv, moo_hv]


# Associative memory:

memory = (dog_hv * bark_hv) + (cat_hv * meow_hv) + (cow_hv * moo_hv);

# Querying memory to search for dog's sound:

findmax(hv -> similarity(memory * dog_hv, hv), sounds)

# Querying memory to search which animals goes "moo":

findmax(hv -> similarity(memory * moo_hv, hv), animals)

# This is a very simple example, but you could think of having a more complex thing going on or
# having more animal that, for example, share sounds.
#
# ### Sequences
#
# **N-grams** represent sequences by encoding the order of elements. This is particularly useful for text processing where word order matters.
#
# Generate hypervectors for all characters in the alphabet

char2hv = Dict(c => BipolarHDV() for c in 'a':'z')
char2hv[' '] = BipolarHDV()

# Encode the phrases using 3-grams

phrases = [
    "the quick brown fox jumps over the lazy dog",
    "the slick grown box bumps under the hazy fog",
    "the thick known cox dumps inter the crazy cog",
    "the brick shown pox lumps enter the glazy jog",
    "the stick blown sox pumps winter the blazy log",
]

ngrams(p, d) = aggregate([d[p[i]] + Π(d[p[i + 1]], 1) + Π(d[p[i + 2]], 2) for i in 1:(length(p) - 2)])

phrases_hvs = [ngrams(p, char2hv) for p in phrases]

# Search for "crazy" in phrases

query = ngrams("crazy", char2hv)
findmax(h -> similarity(query, h), phrases_hvs)

# Great! We correctly found that "crazy" is in phrase 3.
