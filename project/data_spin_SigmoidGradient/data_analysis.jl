using CairoMakie
using CurveFit

function read_data()
    file_names = readdir("project/data_spin_SigmoidGradient")
    file_data = []
    for file in file_names
        match(r"m=(\d+)", file) === nothing && continue
        m = parse(Float64, match(r"m=(\d+)", file).captures[1])
        velocity = parse(Float64, readline("project/data_spin_SigmoidGradient/$file"))
        push!(file_data, (m, velocity))
    end
    sort!(file_data, by=x->x[1])
    x_data = [x[1] for x in file_data]
    y_data = [x[2] for x in file_data]
    return x_data, y_data
end

x_data, y_data = read_data()

function plot_data!(ax::Axis, x_data::Vector{Float64}, y_data::Vector{Float64}, this_label::String)
    scatter!(ax, x_data, y_data, label = this_label)
end

fig = Figure(size=(600, 600))
ax = Axis(fig[1, 1], xlabel = "Depth", ylabel = "Velocity", xscale = log, yscale = log, title = "Velocity v.s. Depth (SigmoidGradient)")
plot_data!(ax, x_data, y_data, "Width = 1.0, h = 30")
fig

# using SurfaceProgrammableMaterial
# eg = SigmoidGradient(2.0, 0.1, -1.0/log(1e-5))
# show_temperature_curve(eg, -5, 5)