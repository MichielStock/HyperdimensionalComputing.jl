# # Encoding data: from raw data to hypervectors
#
# Every HDC application starts with the same question: *how do I turn my data into
# hypervectors?* This tutorial walks through the answer for the four kinds of data you are
# most likely to have -- tokens, sequences, numbers and feature vectors -- and explains which
# encoder to reach for in each case.
#
# It helps to know that `HyperdimensionalComputing.jl` is organised in three layers:
#
# | Layer            | What it does                              | Examples                                   |
# |:-----------------|:------------------------------------------|:-------------------------------------------|
# | **Primitives**   | hypervector in, hypervector out           | `bundle`, `bind`, `shift`, `perturbate`     |
# | **Combinators**  | *collections* of hypervectors in, one out | `multiset`, `ngrams`, `hashtable`, `graph`  |
# | **Encoders**     | **raw data** in, hypervector out          | `encode`, `LevelEncoder`, `RandomProjection`|
#
# The *"Introduction to HDC"* tutorial covers the first two layers. This one is about the third.

using HyperdimensionalComputing

# ## Tokens: one object, one hypervector
#
# The simplest thing you can do is treat an object as an atomic symbol -- a *token* -- and give
# it its own hypervector. That is what [`encode`](@ref) does when you call it without a strategy:
# it hashes the object and uses the hash to seed the vector.

encode(BinaryHV, "cat")

# Two things make this useful. First, it is **deterministic**: the same object always yields the
# same hypervector, in this session or the next, without you having to store a lookup table.

encode(BinaryHV, "cat") == encode(BinaryHV, "cat")

# Second, *different* objects yield **quasi-orthogonal** hypervectors -- they are as unrelated as
# two random vectors, which is exactly what you want from unrelated symbols:

similarity(encode(BinaryHV, "cat"), encode(BinaryHV, "dog"))

# Is that number small? It depends on the metric, and this is a common source of confusion.
# `BinaryHV` uses Jaccard similarity, for which *unrelated* vectors score $1/3$ -- not $0$. Two
# helper functions save you from having to remember which type uses what:

similaritymetric(BinaryHV), chancesimilarity(BinaryHV)

#

similaritymetric(BipolarHV), chancesimilarity(BipolarHV)

# So `0.33` means "unrelated" for a `BinaryHV` and "clearly related" for a `BipolarHV`. Always
# read a similarity against [`chancesimilarity`](@ref) for its type.

# Any hashable object works -- strings, symbols, characters, numbers, tuples:

similarity(encode(BipolarHV, :cat), encode(BipolarHV, 42))

# !!! tip "`HV(x)` is shorthand"
#     `BinaryHV("cat")` is shorthand for `encode(BinaryHV, "cat")`. The shorthand covers tokens
#     only; everything else in this tutorial goes through `encode`. Note that a *number* is not
#     accepted by the shorthand -- `BinaryHV(42)` throws, because it is ambiguous with the
#     dimensionality argument. Use `encode(BinaryHV, 42)` or `BinaryHV(; D = 42)` to say which
#     you meant.
#
# ## Sequences: `KMer` and `NGram` are different things
#
# A string *is* a hashable object, so the token path happily encodes it -- as one atom:

encode(BinaryHV, "ACGTACGT")

# That is often not what you want. `"ACGTACGT"` and `"ACGTACGA"` differ by a single letter, but
# as tokens they are completely unrelated. To encode a sequence *as a sequence*, pass a
# **strategy**. The package ships four, and the first two are easy to confuse, so it is worth
# being precise about the difference.
#
# [`KMer(k)`](@ref) slides a window of length `k` over the sequence and treats **each window as
# one atomic token**: it hashes the whole substring and bundles the results. This is the classic
# k-mer profile from genomics and text processing.

encode(BinaryHV, "ACGTACGT", KMer(3))

# It is exactly equivalent to hashing each window yourself and bundling:

encode(BinaryHV, "ACGT", KMer(3)) == multiset([encode(BinaryHV, w) for w in ["ACG", "CGT"]])

# [`NGram(n)`](@ref) does something genuinely different: it encodes each **symbol**, then binds
# the symbols of a window together with shifts to record their positions, then bundles the
# windows. It is the [`ngrams`](@ref) combinator applied to token-encoded symbols.

encode(BinaryHV, "ACGT", NGram(3)) == ngrams([encode(BinaryHV, c) for c in "ACGT"], 3)

# The two give different hypervectors with different properties:

encode(BinaryHV, "ACGTACGT", KMer(3)) == encode(BinaryHV, "ACGTACGT", NGram(3))

# The practical difference is what *shares structure* with what. Under `KMer`, `"ACG"` and
# `"CGA"` are unrelated tokens -- they merely happen to use the same letters. Under `NGram`, they
# are built from the same three symbol hypervectors, combined in a different order, so they
# retain a relationship. Pick `KMer` when the window is the meaningful unit (k-mer profiles,
# character n-gram fingerprints); pick `NGram` when the symbols are meaningful and you want
# position-aware composition.
#
# The remaining two strategies cover the extremes of "does order matter?":

seq = [encode(BipolarHV, c) for c in "ACGT"]
encode(BipolarHV, "ACGT", Sequence()) == bundlesequence(seq)      # order matters

#

encode(BipolarHV, "ACGT", BagOfSymbols()) == multiset(seq)        # order does not

# ## Example: recognising languages from character k-mers
#
# Here is what k-mer encoding buys you. Every language has a characteristic distribution of
# short character sequences -- `"th"` and `"ing"` are common in English, `"ij"` and `"en"` in
# Dutch. We never tell the computer any of this: we just encode each text as a k-mer profile and
# let similarity do the work.
#
# Our corpus is the opening of the Wikipedia article on DNA in seven languages (including
# [West Flemish](https://en.wikipedia.org/wiki/West_Flemish), a regional variant of Dutch):

texts = Dict(
    "english" => "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates, nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.",
    "simple english" => "DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic code of living organisms. This includes animals, plants, protists, archaea and bacteria. It is made up of two polynucleotide chains in a double helix. DNA is in each cell in the organism and tells cells what proteins to make. Mostly, these proteins are enzymes. DNA is inherited by children from their parents.",
    "dutch" => "Desoxyribonucleinezuur, beter bekend als DNA, is het biologische macromolecuul dat in alle levende cellen de basis vormt van erfelijkheid. DNA is een zeer lang polymeer, en bevat de genetische instructies voor de ontwikkeling, het functioneren, de groei en de voortplanting van alle bekende organismen en vele virussen. DNA heeft een ingewikkelde chemische structuur.",
    "west-flemish" => "De primaire structuur van DNA is de volgorde van de nucleotiedn. Een nucleotiede es ip zyn beurt ipgebouwd uut 3 bouwsteenn: de fosfoatgroep, nen vuufoekign suker, en een boase. Der zyn 4 soortn boasn: twee purines en twee pyrimidines. Der komn ook vele modificoaties voor.",
    "german" => "Desoxyribonukleinsaure, meist kurz als DNA, seltener auch als DNS abgekurzt, ist eine aus unterschiedlichen Desoxyribonukleotiden aufgebaute Nukleinsaure. Sie tragt die Erbinformation bei allen Lebewesen und den DNA-Viren. Das langkettige Polynukleotid enthalt in Abschnitten von Genen besondere Abfolgen seiner Nukleotide.",
    "french" => "L'acide desoxyribonucleique, ou ADN, est une macromolecule biologique presente dans presque toutes les cellules ainsi que chez de nombreux virus. L'ADN contient toute l'information genetique, appelee genome, permettant le developpement, le fonctionnement et la reproduction des etres vivants.",
    "spanish" => "El acido desoxirribonucleico, conocido por las siglas ADN, es un acido nucleico que contiene las instrucciones geneticas fundamentales para el desarrollo, funcionamiento y reproduccion de todos los seres vivos y algunos virus; tambien es responsable de la transmision hereditaria.",
);

# A little cleaning first -- lowercase, and drop punctuation so that we encode letters rather
# than typography:

clean(s) = replace(lowercase(s), r"[^\p{L}\p{N}\s]" => "", r"\s+" => " ")

# And now the whole model. One line per language:

profiles = Dict(lang => encode(BinaryHV, clean(text), KMer(3)) for (lang, text) in texts);

# That is the entire "training": no gradients, no iterations, one pass over each text. Let us
# rank every pair of languages by the similarity of their profiles:

ranked = sort(
    [
        (similarity(profiles[a], profiles[b]), a, b)
            for a in keys(texts) for b in keys(texts) if a < b
    ];
    rev = true,
)
first(ranked, 5)

# The top of the list recovers real linguistic structure. English and Simple English are the most
# similar pair; Dutch and West Flemish come next, followed by the Romance pair (French/Spanish)
# and the Germanic pair (Dutch/German). Nothing about language families was encoded anywhere --
# it falls out of character statistics alone.
#
# And the least similar pairs are the ones that cross family boundaries:

last(ranked, 3)

# !!! note "This is the k-mer workflow in general"
#     Replace the texts with DNA reads and you have a sequence classifier; the code does not
#     change. `encode(HV, sequence, KMer(k))` is the workhorse of HDC in genomics, and the
#     reason `KMer` is a strategy in its own right rather than a special case of `NGram`.
#
# ## Numbers: `LevelEncoder`
#
# Tokens are the wrong tool for numbers. Hashing `3.0` and `3.1` gives two unrelated
# hypervectors, throwing away the one thing that matters about numbers: that nearby values are
# nearly the same. [`LevelEncoder`](@ref) fixes that by building a *ladder* of hypervectors in
# which neighbouring rungs are similar and distant rungs are not.

lvl = LevelEncoder(BipolarHV, 0:0.5:10)

# Encoding is done through the encoder object:

similarity(encode(lvl, 3.0), encode(lvl, 3.5))     # neighbours: similar

#

similarity(encode(lvl, 3.0), encode(lvl, 9.0))     # far apart: quasi-orthogonal

# Because the ladder lives *inside* the encoder, everything encoded through the same `lvl` is
# mutually comparable -- and [`decode`](@ref) can map a hypervector back to a number:

decode(lvl, encode(lvl, 3.0))

# This is what makes `LevelEncoder` a **stateful** encoder: it is not a recipe you can re-derive,
# it is shared state that encoding and decoding must agree on. The *Iris dataset* tutorial uses
# it to encode flower measurements.
#
# ## Feature vectors: `RandomProjection`
#
# Finally, data that is already a vector of numbers -- measurements, embeddings, pixel values --
# is handled by [`RandomProjection`](@ref), which multiplies your `d`-dimensional input by a
# fixed random `D × d` matrix and applies a nonlinearity. Distances are approximately preserved
# (the Johnson–Lindenstrauss lemma), so similar inputs give similar hypervectors.

rp = RandomProjection(BipolarHV, 3; seed = 1)

# Colours are a nice low-dimensional example: RGB triples where we already know which ones
# *should* be similar.

teal = encode(rp, [0.09, 0.57, 0.6])
sky = encode(rp, [0.02, 0.65, 0.9])
orange = encode(rp, [0.99, 0.39, 0.04])

similarity(teal, sky)     # both blue-green: similar

#

similarity(teal, orange)  # opposite ends of the spectrum: dissimilar

# Like `LevelEncoder`, this is stateful: the projection matrix is drawn once, and only vectors
# encoded through the *same* `rp` can be compared. The *Colours* tutorial explores this in depth,
# including how to decode back and how the different matrix types behave.
#
# ## Adding your own strategy
#
# The sequence strategies are deliberately thin -- each is a struct plus one `encode` method --
# so adding your own takes about four lines. Suppose you want to encode only every second symbol:

struct EveryOther <: AbstractEncoding end

function HyperdimensionalComputing.encode(HV::Type{<:AbstractHV}, x, ::EveryOther; kwargs...)
    return multiset([encode(HV, s; kwargs...) for s in collect(x)[1:2:end]])
end

encode(BinaryHV, "ACGTACGT", EveryOther()) == encode(BinaryHV, "AGAG", BagOfSymbols())

# That is the whole extension point: subtype [`AbstractEncoding`](@ref), add one method, and your
# strategy composes with everything else in the package.
#
# ## Summary
#
# | Your data              | Use                                    |
# |:-----------------------|:---------------------------------------|
# | symbols, categories    | `encode(HV, x)`                        |
# | sequences (windows)    | `encode(HV, x, KMer(k))`               |
# | sequences (symbols)    | `encode(HV, x, NGram(n))`              |
# | ordered / unordered    | `Sequence()` / `BagOfSymbols()`        |
# | numbers                | `LevelEncoder`                         |
# | feature vectors        | `RandomProjection`                     |
