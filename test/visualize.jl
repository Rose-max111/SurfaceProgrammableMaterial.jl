using Test, SurfaceProgrammableMaterial, CairoMakie
using SurfaceProgrammableMaterial:evaluate_temperature

@testset "visualize" begin
    sa = SimulatedAnnealingHamiltonian(16, 60, CellularAutomata1D(110))
    # fluctuation exp(-1/T) ~ ϵ
    epsilon = 1/nspin(sa)/100
    # eg = ExponentialGradient(5.0, 1.0, -1/log(epsilon))
    # eg = SigmoidGradient(2.0, 0.1, -1/log(1e-5))
    eg = SigmoidGradient(2.0, 0.1, 1e-4)
    @show SurfaceProgrammableMaterial.cutoff_distance(eg)
    @test show_temperature_matrix(eg, sa, 1.5) isa CairoMakie.Figure

    # tracking
    tracker = SAStateTracker()
    annealing_time = 100
    r = SARuntime(Float64, sa, 10)
    track_equilibration_pulse!(r, eg, annealing_time; tracker, reverse_direction=false)
    @test animate_tracker(sa, tracker, 1; step=1, cutoffT=SurfaceProgrammableMaterial.lowest_temperature(eg)*2) === nothing

    for i in sa.n+1:nspin(sa)
        local_energy = SurfaceProgrammableMaterial.unsafe_energy(sa, r.state, i, 1)
        if local_energy != 0
            @show i, CartesianIndices((sa.n, sa.m))[i], local_energy
        end
    end
    @show SurfaceProgrammableMaterial.unsafe_energy(sa, tracker.state[35], 149, 1)
    @show SurfaceProgrammableMaterial.unsafe_energy(sa, tracker.state[36], 149, 1)
end

tg = ExponentialGradient(30.0, 1.0, 0)
SurfaceProgrammableMaterial.cutoff_distance(tg)
show_effective_temperature_curve(tg, -40, 10)