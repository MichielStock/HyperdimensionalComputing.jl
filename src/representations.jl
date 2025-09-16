#=
Representing the hypervectors using pretty printing
and a custom plotting recipe
=#

using StatsBase, UnicodePlots

function unicodeheatmap(hv::AbstractHV)
    N = length(hv)
    nsq = floor(Int, sqrt(N))
    Np = nsq^2
    return heatmap(reshape(hv[1:Np], (nsq, nsq)))
end

function Base.show(io::IO, mime::MIME"text/plain", hv::AbstractHV)
    println(io, "$(length(hv))-element $(typeof(hv))")
    println(io, "mean ± std : $(mean(hv)) ± $(std(hv))")
    println(io, boxplot(hv.v))
    println(io, unicodeheatmap(hv))
end

function Base.show(io::IO, mime::MIME"text/plain", hv::Union{BinaryHV,BipolarHV})
    counts = Dict(e => count(==(e), hv) for e in unique(hv))
    println(io, "$(length(hv))-element $(typeof(hv))")
    println(io, barplot(counts))
    println(io, unicodeheatmap(hv))
end

function Base.show(io::IO, hv::AbstractHV)
    print(io, "$(length(hv))-element $(typeof(hv))")
end