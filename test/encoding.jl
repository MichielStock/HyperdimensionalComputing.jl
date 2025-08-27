@testset "encoding" begin
    @testset "multiset" begin
        v = [
            BinaryHDV(v) for v in [
                    [true, false, false, false, false],
                    [true, true, false, false, false],
                    [true, true, true, false, false],
                    [true, true, true, true, false],
                    [true, true, true, true, true],
                ]
        ]
        @test multiset(v).v == [true, true, true, false, false]
    end
    @testset "multibind" begin
        v = [
            BinaryHDV(v) for v in [
                    [true, false, false, false, false],
                    [true, true, false, false, false],
                    [true, true, true, false, false],
                    [true, true, true, true, false],
                    [true, true, true, true, true],
                ]
        ]
        @test multibind(v).v == [true, false, true, false, true]
    end
    @testset "bundlesequence" begin
        v = [
            BinaryHDV(v) for v in [
                    [true, false, false, false, false],
                    [true, true, false, false, false],
                    [true, true, true, false, false],
                    [true, true, true, true, false],
                    [true, true, true, true, true],
                ]
        ]
        @test bundlesequence(v).v == [true, false, true, false, true]
    end
    @testset "bindsequence" begin
        v = [
            BinaryHDV(v) for v in [
                    [true, false, false, false, false],
                    [true, true, false, false, false],
                    [true, true, true, false, false],
                    [true, true, true, true, false],
                    [true, true, true, true, true],
                ]
        ]
        @test bindsequence(v).v == [true, false, true, false, true]
    end
    @testset "hashtable" begin
    end
    @testset "crossproduct" begin
    end
    @testset "ngrams" begin
    end
    @testset "graph" begin
    end
end
