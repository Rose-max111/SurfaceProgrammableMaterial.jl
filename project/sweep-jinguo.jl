using SurfaceProgrammableMaterial
using CairoMakie
using Random

function main(;
        annealing_time = 2000,
        n::Int=20,
        m::Int=50,
        width=1.0,
        Tmax=2.0,
        epsilon=1e-5,
        seed=42,
    )
    Random.seed!(seed)
    sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
    lowest_temperature = -1/log(epsilon)
    @info "lowest temperature = $lowest_temperature"
    eg = SigmoidGradient(Tmax, lowest_temperature, width)
    tracker = SAStateTracker()

    r = SARuntime(Float64, sa, 1)
    track_equilibration_pulse!(r, eg, annealing_time; tracker, reverse_direction=false)
    animate_tracker(sa, tracker, 1; step=10, cutoffT=eg.low_temperature*2)
    return r
end

r = main(n=20, m=20, Tmax=2.0, width=1.0, epsilon=1e-5, annealing_time=4000, seed=42)

why(r)
