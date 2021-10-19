#=
operations.jl; This file implements operations that can be done on hypervectors to enable them to encode text-based data.
=#


"""
This function takes two hypervectors as input and returns a vector that is dissimilar to both. For bipolar vectors, this is an elementwise multiplication.
"""
function multiply(hv1::Vector{Int}, hv2::Vector{Int})
    return hv1 .* hv2
end


"""
This function takes two hypervectors as input and returns a vector that is dissimilar to both. For binary vectors, this is an X-OR operation.

QUESTION: does this actually work elementwise?
"""
function multiply(hv1::BitVector, hv2::BitVector)
    return hv1 .⊻ hv2
end


"""
This functions aggragates two binary hypervectors to a new one that is similar to both. For bipolar vectors, this is possible by summing the elements at each position and returning their sign.
"""
function add(hv1::Vector{Int}, hv2::Vector{Int})
    return hv1 + hv2
end


""" 
This functions aggragates two binary hypervectors to a new one that is similar to both. For binary vectors (where averaging 0 and 1 is not possible), this is done by randomly picking an element from each at every position. 
This way, two ones return a one, two zeros return a zero, and a one and a zero have a 50% chance of returning either.

DISCLAIMER: this solution to averaging binary vectors is a bit iffy, but might still work.
"""
function add(hv1::BitVector, hv2::BitVector)
    return convert(BitVector, [rand([hv1[x], hv2[x]]) for x in 1:HYPERVECTOR_DIM])
end


"""
This function rotates a bipolar hypervector by a given degree in order to encode its position. 

DISCLAIMER: still to be benchmarked. This might not be the fastest way to do this as circshift creates new vectors and allocates extra memory to a vector that in this case will be instantly discarded.
"""
function rotate(hv::Vector, degree::Int)
    return circshift(hv, -degree)
end


"""
This function rotates a binary hypervector by a given degree in order to encode its position. 
"""
function rotate(hv::BitVector, degree::Int)
    return hv >> -degree
end
