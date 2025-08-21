@testset "inference" begin
    
    @testset "BinaryHV" begin
        x = BinaryHV([true, false, true, true])
        y = BinaryHV([false, false, false, true])

        @test similarity(x, y) ≈ 1 / 3 ≈ sim_jacc(x.v, y.v)
    end

    @testset "GradedHV" begin
        x = GradedHV([0.1, 0.4, 0.6, .8])
        y = GradedHV([0.9, 0.8, 0.1, 0.3])

        @test similarity(x, y) ≈ sim_jacc(x.v, y.v) ≈ dot(x.v, y.v) / sum(xi+yi-xi*yi for (xi, yi) in zip(x,y))

    end

    @testset "BipolarHV" begin
        x = BipolarHV([1, -1, -1, -1])
        y = BipolarHV([-1, -1, 1, -1])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x, y) ≈ dot(xd,yd) / norm(xd) / norm(yd)
    end

    @testset "TernaryHV" begin
        x = TernaryHV([1, -1, -1, -0])
        y = TernaryHV([-1, -0, 1, -1])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd,yd) / norm(xd) / norm(yd)
    end

    @testset "GradedBipolarHV" begin
        x = GradedBipolarHV([0.1, -0.4, 0.6, .8])
        y = GradedBipolarHV([0.9, 0.8, -0.1, -0.3])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd,yd) / norm(xd) / norm(yd)
    end

    @testset "RealHV" begin
        x = RealHV([0.1, -0.4, 0.6, .8])
        y = RealHV([0.9, 0.8, -0.1, -0.3])

        xd = collect(x)
        yd = collect(y)

        @test similarity(x, y) ≈ sim_cos(x.v, y.v) ≈ dot(xd,yd) / norm(xd) / norm(yd)
    end
end