module MakieExt
using CairoMakie, SurfaceProgrammableMaterial

function SurfaceProgrammableMaterial.show_temperature_matrix(tg::TemperatureGradient, sa::SimulatedAnnealingHamiltonian, middle_position::Real)
    tm = zeros(sa.n, sa.m)
    SurfaceProgrammableMaterial.temperature_matrix!(tm, tg, sa, middle_position)
    f = Figure(size = (600, sa.n/sa.m*600))
    ax = Axis(f[1, 1], aspect = sa.m/sa.n, xlabel = "time", ylabel = "space")
    image!(ax, tm')
    return f
end

function energy_mesh(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, ibatch::Int)
    return reshape([i<=sa.n ? 0.0 : SurfaceProgrammableMaterial.unsafe_energy(sa, state, i, ibatch) for i in 1:nspin(sa)], sa.n, sa.m)
end
function SurfaceProgrammableMaterial.animate_tracker(sa::SimulatedAnnealingHamiltonian, tracker::SAStateTracker, ibatch::Int; filename::String=tempname()*".mp4", framerate::Int=24, step::Int=1)
    @assert length(tracker.state) == length(tracker.temperature) > 0
    emesh = Observable(energy_mesh(sa, tracker.state[1], ibatch)')
    temperature = Observable(reshape(tracker.temperature[1][:, ibatch], sa.n, sa.m)')
    f = Figure(size = (600, sa.n/sa.m*1200))
    ttt = Observable(0.0)
    ax_temperature = Axis(f[1, 1], title = @lift("Step = $($ttt)"), aspect = sa.m/sa.n)
    ax_energy = Axis(f[2, 1], title = "State", aspect = sa.m/sa.n)
    heatmap!(ax_temperature, temperature)
    heatmap!(ax_energy, emesh)
    record(f, filename, 1:step:length(tracker.state); framerate) do val
        @info "step = $val"
        ttt[] = val
        emesh[] = energy_mesh(sa, tracker.state[val], ibatch)'
        temperature[] = reshape(tracker.temperature[val][:, ibatch], sa.n, sa.m)'
    end
    @info "Video saved to: $filename"
end
end