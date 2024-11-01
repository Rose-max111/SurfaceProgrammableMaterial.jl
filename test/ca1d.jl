using Test
using SurfaceProgrammableMaterial:automatarule

@testset "automatarule" begin
    # First test rule 110
    @test(automatarule(1, 1, 1, 110) == 0)
    @test(automatarule(1, 1, 0, 110) == 1)
    @test(automatarule(1, 0, 1, 110) == 1)
    @test(automatarule(1, 0, 0, 110) == 0)
    @test(automatarule(0, 1, 1, 110) == 1)
    @test(automatarule(0, 1, 0, 110) == 1)
    @test(automatarule(0, 0, 1, 110) == 1)
    @test(automatarule(0, 0, 0, 110) == 0)

    # Randomly test some rules
    @test(automatarule(1, 1, 1, 30) == 0)
    @test(automatarule(1, 0, 1, 30) == 0)
    @test(automatarule(0, 1, 1, 30) == 1)
end
