using LinearAlgebra, Random

@testset "operations" begin

    hv_types = [BinaryHV, BipolarHV, RealHV, TernaryHV,
                GradedHV, GradedBipolarHV]

    for HV in hv_types

        N = 500

        @testset "operations $HV" begin

            hv1 = HV(N)
            hv2 = HV(N)

            v1 = collect(hv1)
            v2 = collect(hv2)

            @testset "aggregate $HV" begin
                @test aggregate((hv1, hv2)) isa HV
                @test aggregate([hv1, hv2]) isa HV
                @test aggregate((HV(N) for i in 1:5)) isa HV
                @test hv1 + hv2 isa HV
            end

            @testset "bind $HV" begin
                @test bind(hv1, hv2) isa HV
                @test hv1 * hv2 isa HV
            end

            @testset "shift $HV" begin
                @test shift(hv1, 3) ≈ circshift(v1, 3)
                @test shift!(hv2, -8) ≈ circshift(v2, -8)
                @test ρ(hv1, 2) ≈ circshift(v1, 2)
            end

            @testset "perturbate $HV" begin
                @test perturbate(hv1, 10) isa HV
                @test perturbate(hv2, 0.2) isa HV
                hvp = perturbate(hv1, [4, 8])
                @test hvp.v[[1,2,3]] ≈ hv1.v[[1,2,3]]

                m = bitrand(length(hv1))
                hvp = perturbate(hv2, m)
                @test hv2.v[m] != hvp.v[m]
                @test hv2.v[.!m] ≈ hvp.v[.!m]
            end

            # currently not yet a good way of evaluating these
            HV <: Union{TernaryHV,GradedHV,GradedBipolarHV,RealHV} && continue

            @testset "similarity $HV" begin
                N = 10_000
                hv1 = HV(N)
                hv2 = HV(N) 
                hv3 = HV(N) + hv1  # similar to 1 but not to 2
                normalize!(hv3)

                @test !(hv1 ≈ hv2)

                @test !(hv1 == hv2)
                @test (hv3 ≈ hv1)
                @test !(hv3 ≈ hv2)
            end
            
        end

    end
end