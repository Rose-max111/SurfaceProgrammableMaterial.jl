using SurfaceProgrammableMaterial, CairoMakie
using SurfaceProgrammableMaterial: evaluate_temperature

function effective_temp(tg::TemperatureGradient, pos::Real)
    return sqrt(sqrt(evaluate_temperature(tg, pos) * evaluate_temperature(tg, pos-1)) * evaluate_temperature(tg, pos-1))
end
function test(tg::TemperatureGradient, minimum_distance::Real, maximum_distance::Real)
    x = range(minimum_distance, maximum_distance, length=500)
    y = [effective_temp(tg, i) for i in x] # 第i层为结尾的gadget的effective temperature
    y1 = [effective_temp(tg, i+1) / effective_temp(tg, i) for i in x]
    # y2 = [effective_temp_spin(tg, i) for i in x]
    # y3 = [effective_temp_spin(tg, i+1) / effective_temp_spin(tg, i) for i in x]

    # y = [evaluate_temperature(tg, i) for i in x]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "distance", ylabel = "temperature", title = "width=$(tg.width), Tmax=$(tg.amplitude)", yticks=[0.1, 1, 3, 5, 10, round(maximum(y1), digits=3)], xticks=[-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30])
    lines!(ax, x, y, label="effective temp")
    lines!(ax, x, y1, label = "Teff(i+1) / Teff(i)")
    # lines!(ax, x, y2, label = "effective temp spinmodel")
    # lines!(ax, x, y3, label = "Teff(i+1) / Teff(i) spin")
    axislegend(ax,position=:lt)
    ylims!(ax, low=0, high=10)
    # @show minimum(y)
    return fig
end


tg = ExponentialGradient(100.0, 0.95, 0)
test(tg, -30, 30)