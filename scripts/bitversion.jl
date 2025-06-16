#=
Created on 21/04/2023 11:03:20
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

A version of HDC that makes use of bitvectors
for speed
=#

using Random, BenchmarkTools, LinearAlgebra

const N = 10_000

# BITVECTOR version
# -----------------

hdv(N=N) = bitrand(N)

bind(v::BitVector, u::BitVector) = v .⊻ u
bundle(vs::Vector{<:BitVector}) = BitVector(reduce(.+, vs) .> length(vs)/2)

function shift(v::BitVector, k=1)
    u = similar(v)
    u.chunks[1] = v.chunks[1] >> k
    rem = v.chunks[1] << (64 - k)
    @inbounds for i in 2:length(v.chunks)
        vi = v.chunks[i]
        u.chunks[i] = rem + (vi >> k)
        rem = vi << (64 - k)
    end
    return u
end

hamming(u::BitVector, v::BitVector) = mapreduce(p->count_zeros(p[1]⊻p[2]), +, zip(u.chunks, v.chunks))

nones(u::BitVector) = mapreduce(count_ones, +, u.chunks)
tanimoto(u::BitVector, v::BitVector) = dot(u, v) / (nones(u) * nones(v))

@btime hdv()

u, v, q, r = (hdv() for i in 1:4)

hdvset = [u, v, q, r]

@btime bind($u, $v)
@btime bundle($hdvset)
@btime shift($u, k)

@btime hamming($u, $v)
@btime tanimoto($u, $v)

# BOOL VECTORS
# ------------

hdv(N=N) = rand(Bool, N)

bind(v, u) = v .⊻ u
bundle(vs) = reduce(.+, vs) .> length(vs)/2

shift(v, k=1) = circshift(v, k)


hamming(u, v) = mapreduce(p->==(p...), +, zip(u, v))

tanimoto(u, v) = dot(u, v) / (count(u) * count(v))

@btime hdv()

u, v, q, r = (hdv() for i in 1:4)

hdvset = [u, v, q, r]

@btime bind($u, $v)
@btime bundle($hdvset)

@btime shift($v, 1)

@btime hamming($u, $v)
@btime tanimoto($u, $v)


# so bitvector are approx 10 times faster