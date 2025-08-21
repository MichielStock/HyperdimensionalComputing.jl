module HyperdimensionalComputing

using Distances, Random, Distributions, LinearAlgebra

export AbstractHV, BinaryHV, BipolarHV,
    GradedBipolarHV, RealHV, GradedHV, TernaryHV
export aggregate, bind, shift!, shift, ρ, ρ!, perturbate, perturbate!
export sequence_embedding, sequence_embedding!
export compute_1_grams, compute_2_grams, compute_3_grams, compute_4_grams, compute_5_grams, 
        compute_6_grams, compute_7_grams, compute_8_grams
export similarity, sim_cos, sim_jacc, dist_hamming
export train, predict, retrain!

include("types.jl")
include("operations.jl")
#include("encoding.jl")
include("inference.jl")
#include("learning.jl")

end