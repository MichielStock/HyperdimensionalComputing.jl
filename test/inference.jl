@testset "inference" begin

    import HyperdimensionalComputing: sim_cos, sim_jacc

    @testset "BinaryHV" begin
        x = BinaryHV([true, false, true, true])
        y = BinaryHV([false, false, false, true])

        @test similarity(x, y) ≈ 1 / 3 ≈ sim_jacc(x.v, y.v)
        @test similarity(x, y) == δ(x)(y)
    end

    @testset "GradedHV" begin
        x = GradedHV([0.1, 0.4, 0.6, 0.8])
        y = GradedHV([0.9, 0.8, 0.1, 0.3])

        @test similarity(x, y) ≈ sim_jacc(x.v, y.v) ≈ dot(x.v, y.v) / sum(xi + yi - xi * yi for (xi, yi) in zip(x, y))
        @test similarity(x, y) == δ(x)(y)
    end

    @testset "BipolarHV" begin
        x = BipolarHV([1, -1, -1, -1])
        y = BipolarHV([-1, -1, 1, -1])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x, y) ≈ dot(xd, yd) / norm(xd) / norm(yd)
        @test similarity(x, y) == δ(x)(y)
    end

    @testset "TernaryHV" begin
        x = TernaryHV([1, -1, -1, -0])
        y = TernaryHV([-1, -0, 1, -1])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd, yd) / norm(xd) / norm(yd)
        @test similarity(x, y) == δ(x)(y)

        z = TernaryHV{Int8}(vcat(fill(Int8(1), 150), fill(Int8(-1), 150)))
        @test similarity(z, z) ≈ 1
        @test similarity(z.v, z.v; method = :cosine) ≈ 1
    end

    @testset "GradedBipolarHV" begin
        x = GradedBipolarHV([0.1, -0.4, 0.6, 0.8])
        y = GradedBipolarHV([0.9, 0.8, -0.1, -0.3])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd, yd) / norm(xd) / norm(yd)
        @test similarity(x, y) == δ(x)(y)
    end

    @testset "RealHV" begin
        x = RealHV([0.1, -0.4, 0.6, 0.8])
        y = RealHV([0.9, 0.8, -0.1, -0.3])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd, yd) / norm(xd) / norm(yd)
        @test similarity(x, y) == δ(x)(y)
    end

    @testset "δ alias" begin
        @test isconst(HyperdimensionalComputing, :δ)   # regression (TODO §1.7)
        @test δ === similarity
    end

    @testset "similarity metric traits" begin
        @test similaritymetric(BinaryHV) == :jaccard
        @test similaritymetric(GradedHV) == :jaccard
        for HV in [BipolarHV, TernaryHV, RealHV, GradedBipolarHV, FHRR]
            @test similaritymetric(HV) == :cosine
        end
        ## instance forms agree with the type forms
        @test similaritymetric(BinaryHV(; D = 10)) == :jaccard
        @test chancesimilarity(BinaryHV(; D = 10)) == chancesimilarity(BinaryHV)

        ## the documented baseline must match what unrelated hypervectors actually score
        for HV in [BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR]
            measured = mean(similarity(HV(; D = 5_000), HV(; D = 5_000)) for _ in 1:50)
            @test isapprox(measured, chancesimilarity(HV); atol = 0.02)
        end

        ## FHRR's similarity really is cosine, as the docstring claims
        a, b = FHRR(; D = 1000), FHRR(; D = 1000)
        av, bv = collect(a), collect(b)
        @test similarity(a, b) ≈ real(dot(av, bv)) / (norm(av) * norm(bv))
    end

    @testset "Similarity matrix" begin
        levels = LevelEncoder(RealHV, (0, 1), 10; D = 100).levels
        M = similarity(levels)
        @test M isa Matrix
        @test size(M) == (10, 10)
        @test M ≈ M'
    end

    @testset "NN" begin
        x = BinaryHV(trues(5))

        collection = [BinaryHV([i ≤ k for i in 1:5]) for k in 1:5]

        @test nearest_neighbor(x, collection)[2] == 5
        top2 = nearest_neighbor(x, collection, 2)
        @test Set(last.(top2)) == Set([4, 5])

        dict = Dict(zip("abcde", collection))
        @test nearest_neighbor(x, dict)[2] == 'e'
        top2 = nearest_neighbor(x, dict, 2)
        @test Set(last.(top2)) == Set("de")
    end
end
