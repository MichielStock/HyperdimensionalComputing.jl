const n = 100

using Distributions, LinearAlgebra

@testset "vectors" begin
    @testset "BipolarHV" begin

        hdv = BipolarHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Int
        @test hdv[2] isa Int
        @test all(-1 .≤ hdv .≤ 1)
        @test hdv == BipolarHV(hdv.v)
        @test similar(hdv) isa BipolarHV
        @test sum(hdv) == sum(vi for vi in hdv)
    end

    @testset "BinaryHV" begin

        hdv = BinaryHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Bool
        @test hdv[2] isa Bool
        @test hdv == BinaryHV(hdv.v)
        @test similar(hdv) isa BinaryHV
        @test sum(hdv) ≈ sum(hdv.v)
    end

    @testset "TernaryHV" begin

        hdv = TernaryHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Int
        @test hdv[2] isa Int
        @test hdv == TernaryHV(hdv.v)
        @test similar(hdv) isa TernaryHV
        @test sum(hdv) ≈ sum(hdv.v)
    end

    @testset "GradedBipolarHV" begin

        hdv = GradedBipolarHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(-1 .≤ hdv .≤ 1)
        @test hdv == GradedBipolarHV(hdv.v)
        @test similar(hdv) isa GradedBipolarHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(GradedBipolarHV(Float32, n)) <: Float32

        @test GradedBipolarHV(n, 2Beta(10,2)-1) isa GradedBipolarHV

        hv2 = GradedBipolarHV([0.1, 1.12, -0.2, -3.0])
        normalize!(hv2)
        @test all(-1 .≤ hv2 .≤ 1)
        
    end

    @testset "GradedHV" begin

        hdv = GradedHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(0 .≤ hdv .≤ 1)
        @test hdv == GradedHV(hdv.v)
        @test similar(hdv) isa GradedHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(GradedHV(Float32, n)) <: Float32

        @test GradedHV(n, Beta(10,2)) isa GradedHV

        hv2 = GradedHV([0.1, 1.12, -0.2, -3.0])
        normalize!(hv2)
        @test all(0 .≤ hv2 .≤ 1)
    end


    @testset "RealHV" begin

        hdv = RealHV(n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
    
        @test hdv == RealHV(hdv.v)
        @test similar(hdv) isa RealHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(RealHV(Float32, n)) <: Float32
        @test norm(hdv) ≈ norm(hdv.v)
        normalize!(hdv)
    end
end