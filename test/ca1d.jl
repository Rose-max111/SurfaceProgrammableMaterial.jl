using Test
using SurfaceProgrammableMaterial:automatarule
using SurfaceProgrammableMaterial:CellularAutomata1D, simulate

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

@testset "simulate_cellular_automata" begin
    rule = CellularAutomata1D{110}()
    initial_state = Bool.([0, 1, 1, 0, 1, 0, 1, 0])
    nstep = 5
    final_state = simulate(rule, initial_state, nstep)
    # @info final_state[:, 3]
    @test final_state[:, nstep+1] == [1, 0, 0, 1, 1, 0, 0, 0]
    @test final_state[:, 2] == [1, 1, 1, 1, 1, 1, 1, 0]
    @test final_state[:, 3] == [1, 0, 0, 0, 0, 0, 1, 1]
    @test final_state[:, 4] == [1, 0, 0, 0, 0, 1, 1, 0]
end

# 10011000
# 10001111
# 10000110
# 10000011
# 11111110
# 01101010  