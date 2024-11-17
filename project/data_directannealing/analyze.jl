using CairoMakie
using CurveFit

struct Data
    x::Vector{Float64}
    y::Vector{Float64}
end

function read_data()
    files = readdir("project/data_directannealing")
    file_data = []
    for file in files
        m = match(r"t=(\d+)", file)
        if m !== nothing
            t_value = parse(Int, m.captures[1])
            push!(file_data, (t_value, parse(Int, readline("project/data_directannealing/$file"))))
        end
    end
    sort!(file_data, by=x->x[1])

    x_data = [x[1] for x in file_data]
    y_data = [x[2] for x in file_data]

    y_data = 1.0 .- y_data ./ 100000.0
    return Data(x_data, y_data)
end

fitting(data::Data) = curve_fit(LinearFit, data.x, data.y)

data = read_data()

fig = Figure(size = (1200, 600))
ax1 = Axis(fig[1, 1], xscale = log, yscale = log, xlabel = "Sweep Time", ylabel = "Error rate", title = "Error rate v.s. Sweep Time in a 7*7 110 lattice")
scatter!(ax1, data.x, data.y)

ax2 = Axis(fig[1, 2], xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], title = "Each Tangent slope", xlabel = "Tangent index", ylabel = "Slope")
for i in 1:10:length(data.x)
    nxt = min(i + 9, length(data.x))
    data_temp = Data(data.x[i:nxt], data.y[i:nxt])
    fit = fitting(Data(log.(data_temp.x), log.(data_temp.y)))
    scatter!(ax2, [Int((i-1) / 10 + 1)], [fit.coefs[2]])
    tangent_y = fit.coefs[2] .* log.(data_temp.x) .+ fit.coefs[1]
    tangent_x = log.(data_temp.x)
    lines!(ax1, exp.(tangent_x), exp.(tangent_y))
end

fig