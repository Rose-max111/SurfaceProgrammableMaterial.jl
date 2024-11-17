using CUDA, SurfaceProgrammableMaterial

function evaluate_success_prob(tg::TemperatureCollective, 
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

function __main__(typetemp::Type, n::Int, m::Int, nbatch::Int, sweep_time::Int; Tmax=5.0, Tmin=1e-5, input_fixed = true)
    tg = typetemp(Tmax, Tmin)
    flip_group = parallel_scheme(SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110)); input_fixed)
    input_fixed == true && @info "checking fixed input is $(sort!(vcat(flip_group...)) == collect(n+1:n*m))"
    proper_rate = evaluate_success_prob(tg, n, m, nbatch, sweep_time;
                    CUDA_functional = true, flip_scheme = CUDA.cu.(flip_group))
    @show proper_rate
    return proper_rate
end

function __makefilemain__()
    temperature_type = eval(Symbol(ARGS[1]))
    nbatch = parse(Int, ARGS[2])
    n = parse(Int, ARGS[3])
    m = parse(Int, ARGS[4])
    sweep_step = parse(Int, ARGS[7])
    sweep_minimum = parse(Int, ARGS[5])
    sweep_maximum = parse(Int, ARGS[6])
    input_fixed = parse(Bool, ARGS[8])
    cuda_device = parse(Int, ARGS[9])

    CUDA.device!(cuda_device)
    for runt in sweep_minimum:sweep_step:sweep_maximum
        proper_rate = __main__(temperature_type, n, m, nbatch, runt; input_fixed)
        filepath = input_fixed == true ? joinpath(@__DIR__, "data_globalannealing_inputfixed/n=$(n)_m=$(m)_nbatch=$(nbatch)_t=$(t).txt") : joinpath(@__DIR__, "data_globalannealing/n=$(n)_m=$(m)_nbatch=$(nbatch)_t=$(t).txt")
        open(filepath, "w") do file
            println(file, proper_rate)
        end
    end
end

__makefilemain__()
# __main__(SigmoidGradient, 16, 15, 10000, 0.28005882158045375; width=0.6)
