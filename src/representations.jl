#=
Representing the hypervectors using pretty printing
and a custom plotting recipe

See ext/UnicodePlotting.jl for extensions based on UnicodePlotting
=#

# ------------------------------------------------------------------------------------- Compact
# function Base.show(io::IO, hv::TernaryHV)
#     return print(io, "$(typeof(hv))(D=$(length(hv)), μ=$(round(mean(hv))), σ=$(round(std(hv))))")
# end
#
# function Base.show(io::IO, hv::BinaryHV)
#     D = length(hv.v)
#     n_true = count(identity, hv.v)
#     n_false = D - n_true
#     return print(io, "BinaryHV(D=$(D), trues=$(n_true), falses=$(n_false))")
# end
#
# function Base.show(io::IO, hv::BipolarHV)
#     D = length(hv.v)
#     n_true = count(identity, hv.v)
#     n_false = D - n_true
#     print(io, "BipolarHV(D=$(D), positives=$(n_true), negatives=$(n_false))")
#     return
# end
#
# # ------------------------------------------------------------------------------------- REPL
# function Base.show(io::IO, mime::MIME"text/plain", hv::TernaryHV)
#     println(io, "$(length(hv))-element $(typeof(hv)) with μ ± σ = $(round(mean(hv), digits = 3)) ± $(round(std(hv), digits = 3))")
#     println(repr(mime, hv.v'; context=io))
#     # return split(repr(mime, hv.v; context = io), '\n')[2:end] .|> println
# end
#
# function Base.show(io::IO, mime::MIME"text/plain", hv::BinaryHV)
#     counts = Dict(e => count(==(e), hv) for e in unique(hv))
#     println(io, "$(length(hv))-element $(typeof(hv)) with $(counts[1]) trues and $(counts[0]) falses")
#     return split(repr(mime, hv.v; context = io), '\n')[(begin + 1):end] .|> println
# end
#
# function Base.show(io::IO, mime::MIME"text/plain", hv::BipolarHV)
#     counts = Dict(e => count(==(e), hv) for e in unique(hv))
#     println(io, "$(length(hv))-element $(typeof(hv)) with $(counts[1]) positives and $(counts[-1]) negatives")
#     return split(repr(mime, hv.v .* 2 .- 1; context = io), '\n')[(begin + 1):end] .|> println
# end
function Base.show(io::IO, ::MIME"text/plain", hv::AbstractHV)
    # NOTE: Based off https://github.com/JuliaLang/julia/blob/cf40898d56a5b32c6a2e97f61355440df36a7357/base/arrayshow.jl#L363
    # Fast return for empty hypervectors
    if isempty(hv) && (get(io, :compact, false)::Bool || hv isa AbstractHV)
        return println(io, typeof(hv))
    end

    # 1) show summary before setting :compact
    if typeof(hv) == BinaryHV
        counts = Dict(e => count(==(e), hv) for e in unique(hv))
        print(io, "$(length(hv))-element $(typeof(hv)) with $(counts[true]) true and $(counts[false]) false")
    elseif typeof(hv) == BipolarHV
        counts = Dict(e => count(==(e), hv) for e in unique(hv))
        print(io, "$(length(hv))-element $(typeof(hv)) with $(counts[1]) positives and $(counts[-1]) negatives")
    else
        print(io, "$(length(hv))-element $(typeof(hv)) with μ ± σ = $(round(mean(hv), digits = 3)) ± $(round(std(hv), digits = 3))")
    end
    isempty(hv) && return
    print(io, ":")

    # 2) compute new IOContext
    if !haskey(io, :compact) && length(axes(hv, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false)::Bool && eltype(hv) === Method
        io = IOContext(io, :limit => false)
    end

    if get(io, :limit, false)::Bool && displaysize(io)[1] - 4 <= 0
        print(io, " …")
    else
        println(io)
    end

    # 3) update typeinfo
    io = IOContext(io, :typeinfo => eltype(hv))

    # 4) show actual content
    recur_io = IOContext(io, :SHOWN_SET => hv)
    return Base.print_array(recur_io, hv)
end
