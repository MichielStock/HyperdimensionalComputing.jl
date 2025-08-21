#=
inference.jl; This file implements functions to compare two hyperdimensional vectors.
=#

using LinearAlgebra

LinearAlgebra.dot(u::AbstractHV, v::AbstractHV) = dot(u.v, v.v)

LinearAlgebra.dot(u::BipolarHV, v::BipolarHV) = 4dot(u.v, v.v) - 2sum(u.v) - 2sum(v.v) + length(u)

sim_cos(u::AbstractVector, v::AbstractVector) = dot(u, v) / (norm(u) * norm(v))

sim_jacc(u::AbstractVector, v::AbstractVector) = dot(u, v) / sum(ui+vi-ui*vi for (ui, vi) in zip(u, v))

dist_hamming(u::AbstractVector, v::AbstractVector) = sum(abs(ui-vi) for (ui, vi) in zip(u, v))

similarity(x::BipolarHV, y::BipolarHV) = sim_cos(x, y)
similarity(x::TernaryHV, y::TernaryHV) = sim_cos(x, y)
similarity(x::GradedBipolarHV, y::GradedBipolarHV) = sim_cos(x, y)
similarity(x::RealHV, y::RealHV) = sim_cos(x, y)

similarity(x::BinaryHV, y::BinaryHV) = sim_jacc(x, y)
similarity(x::GradedHV, y::GradedHV) = sim_jacc(x, y)

