using SurfaceProgrammableMaterial, CairoMakie
using SurfaceProgrammableMaterial: evaluate_temperature

function draw!(ax, tg::TemperatureGradient, minimum_distance::Real, maximum_distance::Real, label::String)
    x = range(minimum_distance, maximum_distance, length=500)
    y = [evaluate_temperature(tg, i) for i in x]
    lines!(ax, x, y, label = label)
    return ax
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Distance", ylabel = "Temperature", xticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], title = "Sigmoid Temperature Gradient")
# draw!(ax, SigmoidGradient(2.0, 1.0, -1 / log(1e-5)), -10, 10, "Width = 1.0")
# draw!(ax, SigmoidGradient(2.0, 0.6, -1 / log(1e-5)), -10, 10, "Width = 0.6")
# draw!(ax, SigmoidGradient(2.0, 0.3, -1 / log(1e-5)), -10, 10, "Width = 0.3")
# draw!(ax, SigmoidGradient(2.0, 0.1, -1 / log(1e-5)), -10, 10, "Width = 0.1")
# draw!(ax, SigmoidGradient(2.0, 0.06, -1 / log(1e-5)), -10, 10, "Width = 0.06")
draw!(ax, SigmoidGradient(30.0, 1.0, -1 / log(1e-5)), -10, 10, "Width = 1.0")

axislegend(ax)
fig