using Test, SurfaceProgrammableMaterial, CairoMakie

@testset "visualize" begin
    sa = SimulatedAnnealingHamiltonian(20, 50, CellularAutomata1D(110))
    eg = ExponentialGradient(5.0, 1.0, 1e-5)
    @test show_temperature_matrix(eg, sa, 1.5) isa CairoMakie.Figure

    # tracking
    tracker = SAStateTracker()
    state = random_state(sa, 10)
    annealing_time = 2000
    r = SARuntime(Float64, sa, 10)
    track_equilibration_pulse!(r, eg, annealing_time; tracker, reverse_direction=true)
    @test animate_tracker(sa, tracker, 1; step=10) === nothing
end
