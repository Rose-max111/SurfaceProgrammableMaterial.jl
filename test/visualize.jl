using Test, SurfaceProgrammableMaterial, CairoMakie

@testset "visualize" begin
    sa = SimulatedAnnealingHamiltonian(100, 200, CellularAutomata1D(110))
    eg = ExponentialGradient(1.0, 1.0, 1e-5)
    @test show_temperature_matrix(eg, sa, 1.5) isa CairoMakie.Figure
end
