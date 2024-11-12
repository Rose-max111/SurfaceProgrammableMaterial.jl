using CUDA, SurfaceProgrammableMaterial

function evaluate_success_prob(tg::TemperatureGradient, 
    n::Int, m::Int, nbatch::Int, sweep_time::Int;
    CUDA_functional = false,
    flip_scheme = 1:(n*m))

    sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
    r = SARuntime(Float64, sa, nbatch)
    CUDA_functional == true ? (r = CUDA.cu(r)) : nothing

    track_equilibration_pulse!(r, tg, sweep_time; flip_scheme)
    state_energy = CUDA_functional !== false ? energy(sa, Array(r.state)) : energy(sa, r.state)
    success = count(x -> x == 0, state_energy)

    return success / nbatch
end

function __main__(typetemp::Type, n::Int, m::Int, nbatch::Int, moving_velocity::Float64; Tmax=2.0, epsilon=1e-5, width=1.0)
    lowest_temperature = -1/log(epsilon)
    tg = typetemp(Tmax, width, lowest_temperature)
    proper_rate = evaluate_success_prob(tg, n, m, nbatch, Int(ceil((2*SurfaceProgrammableMaterial.cutoff_distance(tg) + m) / moving_velocity));
                    CUDA_functional = true, flip_scheme = CUDA.cu.(parallel_scheme(SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110)))))
    @show proper_rate
    return proper_rate
end

function __makefilemain__()
    temperature_type = eval(Symbol(ARGS[1]))
    nbatch = parse(Int, ARGS[2])
    n = parse(Int, ARGS[3])
    m = parse(Int, ARGS[4])
    width = parse(Float64, ARGS[5])
    num_example = parse(Int, ARGS[6])
    v_maximum = parse(Float64, ARGS[7])
    v_minimum = parse(Float64, ARGS[8])
    cuda_device = parse(Int, ARGS[9])

    CUDA.device!(cuda_device)
    for v in LinRange(v_maximum, v_minimum, num_example)
        proper_rate = __main__(temperature_type, n, m, nbatch, v; width)
        filepath = joinpath(@__DIR__, "data_error_vs_sweep_$(temperature_type)/n=$(n)_m=$(m)_width=$(width)_nbatch=$(nbatch)_v=$(v).txt")
        open(filepath, "w") do file
            println(file, proper_rate)
        end
    end
end

__makefilemain__()
# __main__(SigmoidGradient, 16, 15, 10000, 0.28005882158045375; width=0.6)
