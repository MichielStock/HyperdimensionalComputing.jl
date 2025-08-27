module HyperdimensionalComputing

using Distances

export AbstractHDV, BinaryHDV, BipolarHDV,
    GradedBipolarHDV, RealHDV, GradedHDV
export offsetcombine, offsetcombine!
export aggregate, aggregate!, aggregatewith!, bind, bind!, Π, Π!, resetoffset!
export similarity, jacc_sim, cos_sim
export train, predict, retrain!

include("vectors.jl")
include("operations.jl")
include("encoding.jl")
export multiset,
    multibind,
    bundlesequence,
    bindsequence,
    hashtable,
    crossproduct,
    ngrams,
    graph
include("inference.jl")
include("learning.jl")

end
