using CairoMakie
using CurveFit

function read_data()
    file_names = readdir("project/data_error_vs_sweep_SigmoidGradient")
    file_data = []
    for file in file_names
        velocity = match(r"v=(\d+\.\d+)", file)
        if (velocity === nothing) == false
            velocity = parse(Float64, velocity.captures[1])
            error_rate = 1 - parse(Float64, readline("project/data_error_vs_sweep_SigmoidGradient/$file"))
            push!(file_data, (velocity, error_rate))
        end
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
ax = Axis(fig[1, 1], xlabel = "Velocity", ylabel = "Error rate", xscale = log, yscale = log, title = "Error rate v.s. Velocity on a 30*70 lattice(SigmoidGradient)")
plot_data!(ax, x_data, y_data, "n=30, m=70, width=1.0")
fig

# using SurfaceProgrammableMaterial
# eg = SigmoidGradient(2.0, 0.1, -1.0/log(1e-5))
# show_temperature_curve(eg, -5, 5)