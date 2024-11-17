using CairoMakie
using CurveFit

struct Data
    x::Vector{Float64}
    y::Vector{Float64}
end
Data(x::Vector{Int}, y::Vector{Float64}) = Data(Float64.(x), y)

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
    return Data(x_data, y_data)
end

function power_vs_width(widths::Vector{Float64})
    y_data = Vector{Float64}()
    for expect_width in widths
        data = read_data(expect_width)
        fit = fitting(Data(log.(data.x), log.(data.y)))
        power = fit.coefs[2]
        push!(y_data, power)
    end
    return Data(widths, y_data)
end

fitting(data::Data) = curve_fit(LinearFit, data.x, data.y)

function plot_data!(ax::Axis, data::Data, this_label::String)
    scatter!(ax, data.x, data.y, label = this_label)
end

my_widths = [1.3, 1.0, 0.8, 0.6, 0.3, 0.2, 0.1, 0.06, 0.03]
data_power_width = power_vs_width(my_widths)

fig = Figure(size=(1200, 600))
ax2 = Axis(fig[1, 2], xlabel = "width", ylabel = "slope", title = "slope v.s. width")
ax1 = Axis(fig[1, 1], xscale = log, yscale = log, title = "moving velocity v.s. depth(SigmoidGradient)", xlabel = "depth", ylabel = "velocity",
    # xticks = ([exp(i) for i in 2:5], ["exp($i)" for i in 2:5]),
    # yticks = ([exp(i) for i in -1:5], ["exp($i)" for i in -1:5]),
    aspect = 0.75
)
for id in 1:length(my_widths)
    data = read_data(my_widths[id])
    scatter!(ax1, data.x, data.y, label = "width=$(my_widths[id])")
    scatter!(ax2, [my_widths[id]], [data_power_width.y[id]])
end


axislegend(ax1; position = :lb)
fig

# using SurfaceProgrammableMaterial
# eg = SigmoidGradient(2.0, 0.1, -1.0/log(1e-5))
# show_temperature_curve(eg, -5, 5)