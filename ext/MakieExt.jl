module MakieExt
using CairoMakie, SurfaceProgrammableMaterial
using SurfaceProgrammableMaterial: evaluate_temperature

function SurfaceProgrammableMaterial.show_temperature_matrix(tg::TemperatureGradient, sa::SimulatedAnnealingHamiltonian, middle_position::Real)
    tm = zeros(sa.n, sa.m)
    SurfaceProgrammableMaterial.temperature_matrix!(tm, tg, 1:sa.m, middle_position)
    f = Figure(size = (600, sa.n/sa.m*600))
    ax = Axis(f[1, 1], aspect = sa.m/sa.n, xlabel = "time", ylabel = "space")
    image!(ax, tm')
    return f
end

function energy_mesh(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, ibatch::Int)
    return reshape([i<=sa.n ? 0.0 : SurfaceProgrammableMaterial.unsafe_energy(sa, state, i, ibatch) for i in 1:nspin(sa)], sa.n, sa.m)
end
function SurfaceProgrammableMaterial.animate_tracker(sa::SimulatedAnnealingHamiltonian, tracker::SAStateTracker, ibatch::Int;
        filename::String=tempname()*".mp4",
        framerate::Int=24,
        step::Int=1,
        cutoffT::Real
    )
    @assert length(tracker.state) == length(tracker.temperature) > 0
    emesh = Observable(energy_mesh(sa, tracker.state[1], ibatch)')
    temperature = Observable(reshape(tracker.temperature[1][:, ibatch], sa.n, sa.m)')
    f = Figure(size = (600, sa.n/sa.m*1200))
    ttt = Observable(0.0)
    ax_temperature = Axis(f[1, 1], title = @lift("Step = $($ttt)"), aspect = sa.m/sa.n)
    ax_energy = Axis(f[2, 1], title = "State", aspect = sa.m/sa.n)
    heatmap!(ax_temperature, temperature, colorrange=(cutoffT, maximum(maximum.(tracker.temperature))), colorscale=log10)
    contour!(ax_temperature, temperature, levels=[cutoffT], color=:red)
    heatmap!(ax_energy, emesh)
    record(f, filename, 1:step:length(tracker.state); framerate) do val
        ttt[] = val
        emesh[] = energy_mesh(sa, tracker.state[val], ibatch)'
        temperature[] = reshape(tracker.temperature[val][:, ibatch], sa.n, sa.m)'
    end
    @info "Video saved to: $filename"
end

function SurfaceProgrammableMaterial.show_temperature_curve(tg::TemperatureGradient, minimum_distance::Real, maximum_distance::Real)
    x = range(minimum_distance, maximum_distance, length=500)
    # y = [sqrt(sqrt(evaluate_temperature(tg, i) * evaluate_temperature(tg, i-1)) * evaluate_temperature(tg, i-1)) for i in x]

    y = [evaluate_temperature(tg, i) for i in x]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "distance", ylabel = "temperature")
    lines!(ax, x, y)
    # @show minimum(y)
    return fig
end

function effective_temp(tg::TemperatureGradient, pos::Real)
    return sqrt(sqrt(evaluate_temperature(tg, pos) * evaluate_temperature(tg, pos-1)) * evaluate_temperature(tg, pos-1))
end

function effective_temp_spin(tg::TemperatureGradient, pos::Real)
    return sqrt(evaluate_temperature(tg, pos) * evaluate_temperature(tg, pos-1))
end

function SurfaceProgrammableMaterial.show_effective_temperature_curve(tg::TemperatureGradient, minimum_distance::Real, maximum_distance::Real)
    x = range(minimum_distance, maximum_distance, length=500)
    y = [min(10,effective_temp(tg, i)) for i in x] # 第i层为结尾的gadget的effective temperature
    y1 = [effective_temp(tg, i+1) / effective_temp(tg, i) for i in x]
    y2 = [effective_temp_spin(tg, i) for i in x]
    y3 = [effective_temp_spin(tg, i+1) / effective_temp_spin(tg, i) for i in x]

    # y = [evaluate_temperature(tg, i) for i in x]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "distance", ylabel = "temperature")
    lines!(ax, x, y, label="effective temp toymodel")
    lines!(ax, x, y1, label = "Teff(i+1) / Teff(i) toy")
    # lines!(ax, x, y2, label = "effective temp spinmodel")
    # lines!(ax, x, y3, label = "Teff(i+1) / Teff(i) spin")
    axislegend(ax,position=:lt)
    # @show minimum(y)
    return fig
end


end