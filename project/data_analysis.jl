using CairoMakie
using CurveFit

function read_data(expect_width::Float64)
    file_names = readdir("project/data_SigmoidGradient")
    file_data = []
    for file in file_names
        width = parse(Float64, match(r"width=(\d+\.\d+)", file).captures[1])
        if width == expect_width
            m_value = parse(Int, match(r"m=(\d+)", file).captures[1])
            velocity = parse(Float64, readline("project/data_SigmoidGradient/$file"))
            push!(file_data, (m_value, velocity))
        end
    end
    sort!(file_data, by=x->x[1])
    x_data = [x[1] for x in file_data]
    y_data = [x[2] for x in file_data]
    return x_data, y_data
end
function plot_data!(ax::Axis, x_data::Vector{Float64}, y_data::Vector{Float64}, this_label::String)
    scatter!(ax, x_data, y_data, label = this_label)
    # fit_y = curve_fit(LinearFit, Float64.(x_data), y_data)
    # lines!(ax, x_data, fit_y.(Float64.(x_data)))
end

fig = Figure(size=(600, 600))
ax = Axis(fig[1, 1], xscale = log, yscale = log, title = "moving velocity v.s. depth(SigmoidGradient)", xlabel = "depth", ylabel = "1 / velocity",
    # xticks = ([exp(i) for i in 2:5], ["exp($i)" for i in 2:5]),
    # yticks = ([exp(i) for i in -1:5], ["exp($i)" for i in -1:5]),
    aspect = 0.75
)
for expect_width in [1.0, 0.6, 0.3, 0.1, 0.06]
    x_data, y_data = read_data(expect_width)
    y_data = 1.0 ./ y_data
    x_data = Float64.(x_data)
    plot_data!(ax, x_data, y_data, "Width = $expect_width")
end
axislegend(ax; position = :lt)
fig

# using SurfaceProgrammableMaterial
# eg = SigmoidGradient(2.0, 0.1, -1.0/log(1e-5))
# show_temperature_curve(eg, -5, 5)