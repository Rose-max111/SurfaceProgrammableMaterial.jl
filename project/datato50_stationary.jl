using CUDA, SurfaceProgrammableMaterial

function evaluate_p_percentage_velocity(tg::TemperatureGradient, 
    n::Int, m::Int, nbatch::Int, p::Float64;
    CUDA_functional = false,
    flip_scheme = 1:(n*m))
    sweep_time = 0
    maxtry = 2
    while true
        sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
        r = SARuntime(Float64, sa, nbatch)
        CUDA_functional == true ? (r = CUDA.cu(r)) : nothing

        track_equilibration_pulse!(r, tg, maxtry; flip_scheme)
        state_energy = CUDA_functional !== false ? energy(sa, Array(r.state)) : energy(sa, r.state)
        success = count(x -> x == 0, state_energy)
    
        @info "stage 1, sweept time = $(maxtry), success = $(success / nbatch)"
        success / nbatch >= p ? break : maxtry *= 2
    end
    sweep_time = maxtry / 2
    maxtry /= 4
    while maxtry >= 1
        sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
        r = SARuntime(Float64, sa, nbatch)
        CUDA_functional == true ? (r = CUDA.cu(r)) : nothing

        track_equilibration_pulse!(r, tg, Int(sweep_time + maxtry); flip_scheme)
        state_energy = CUDA_functional !== false ? energy(sa, Array(r.state)) : energy(sa, r.state)
        success = count(x -> x == 0, state_energy)
            
        @info "stage 2, sweept time = $(sweep_time + maxtry), success = $(success / nbatch)"

        success / nbatch < p ? sweep_time += maxtry : nothing
        maxtry /= 2
    end
    final_sweep_time = sweep_time + 1
    return final_sweep_time
end

# epsilon determines the lowest_temperature
# Tmax denote the highest_temperature
function __main__(typetemp::Type, n::Int, m::Int, nbatch::Int; Tmax=0.1, base=0.8, input_fixed = true)
    tg = typetemp(Tmax, base)
    flip_group = parallel_scheme(SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110)); input_fixed)
    @show sort!(vcat(flip_group...))
    input_fixed == true && @assert sort!(vcat(flip_group...)) == collect(n+1:n*m) "flip_group is not correct"
    CUDA.functional() == false && return evaluate_p_percentage_velocity(tg, n, m, nbatch, 0.50; CUDA_functional = false, flip_scheme = (flip_group))
    sweep_time = evaluate_p_percentage_velocity(tg, n, m, nbatch, 0.50; 
                    CUDA_functional = true, flip_scheme = CUDA.cu.(flip_group))
    @show sweep_time
end

function __makefilemain__()
    temperature_type = eval(Symbol(ARGS[1]))
    n = parse(Int, ARGS[2])
    nbatch = parse(Int, ARGS[3])
    base = parse(Float64, ARGS[4])
    m_step = parse(Int, ARGS[5])
    m_minimum = parse(Int, ARGS[6])
    m_maximum = parse(Int, ARGS[7])
    input_fixed = parse(Bool, ARGS[8])
    cuda_device = parse(Int, ARGS[9])

    @show input_fixed

    CUDA.device!(cuda_device)
    for m in m_minimum:m_step:m_maximum
        sweep_time = __main__(temperature_type, n, m, nbatch; base, input_fixed)
        filepath = input_fixed == true ? joinpath(@__DIR__, "data_$(temperature_type)_inputfixed/n=$(n)_m=$(m)_base=$(base)_nbatch=$(nbatch).txt") : joinpath(@__DIR__, "data_$(temperature_type)/n=$(n)_m=$(m)_base=$(base)_nbatch=$(nbatch).txt")
        open(filepath, "w") do file
            println(file, sweep_time)
        end
    end
end

__main__(StationaryExponentialGradient, 10, 10, 1000)

# __makefilemain__()

# CUDA.device!(1)
# __main__(SigmoidGradient, 20, 20, 10000)