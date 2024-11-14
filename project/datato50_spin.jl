using CUDA, SurfaceProgrammableMaterial

function evaluate_p_percentage_velocity(tg::TemperatureGradient, 
    n::Int, m::Int, nbatch::Int, p::Float64;
    CUDA_functional = false,
    flip_scheme = 1:(n*m))
    sweep_time = 0
    maxtry = 2
    sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
    model = spin_model_construction(Float64, sa)
    while true
        r = SpinSARuntime(Float64, sa, nbatch)
        CUDA_functional == true ? (r = CUDA.cu(r)) : nothing

        @info "begin annealing"
        track_equilibration_pulse!(r, tg, maxtry; flip_scheme)
        @info "finish annealing"
        state_energy = energy(model, r.state)
        @show state_energy[1]
        success = count(x -> x == -11 * n * (m-1), state_energy)
    
        @info "stage 1, sweept time = $(maxtry), success = $(success / nbatch)"
        success / nbatch >= p ? break : maxtry *= 2
    end
    sweep_time = maxtry / 2
    maxtry /= 4
    while maxtry >= 1
        r = SpinSARuntime(Float64, sa, nbatch)
        CUDA_functional == true ? (r = CUDA.cu(r)) : nothing

        track_equilibration_pulse!(r, tg, Int(sweep_time + maxtry); flip_scheme)
        state_energy = energy(model, r.state)
        success = count(x -> x == -11 * n * (m-1), state_energy)
            
        @info "stage 2, sweept time = $(sweep_time + maxtry), success = $(success / nbatch)"

        success / nbatch < p ? sweep_time += maxtry : nothing
        maxtry /= 2
    end
    final_sweep_time = sweep_time + 1
    return (2*SurfaceProgrammableMaterial.cutoff_distance(tg) + 2 * m - 1) / (final_sweep_time - 1)
end

# epsilon determines the lowest_temperature
# Tmax denote the highest_temperature
function __main__(typetemp::Type, n::Int, m::Int, nbatch::Int; Tmax=20.0, epsilon=1e-5, width=1.0)
    lowest_temperature = -1/log(epsilon)
    tg = typetemp(Tmax, width, lowest_temperature)
    sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
    r = SpinSARuntime(Float64, sa, 1) # used for construct parallel flip scheme
    moving_speed = evaluate_p_percentage_velocity(tg, n, m, nbatch, 0.50; 
                    CUDA_functional = true, flip_scheme = CUDA.cu.(parallel_scheme(r)))
    @show moving_speed
end

function __makefilemain__()
    temperature_type = eval(Symbol(ARGS[1]))
    n = parse(Int, ARGS[2])
    nbatch = parse(Int, ARGS[3])
    width = parse(Float64, ARGS[4])
    m_step = parse(Int, ARGS[5])
    m_minimum = parse(Int, ARGS[6])
    m_maximum = parse(Int, ARGS[7])
    cuda_device = parse(Int, ARGS[8])

    CUDA.device!(cuda_device)
    for m in m_minimum:m_step:m_maximum
        velocity = __main__(temperature_type, n, m, nbatch; width)
        filepath = joinpath(@__DIR__, "data_spin_$(temperature_type)/n=$(n)_m=$(m)_width=$(width)_nbatch=$(nbatch).txt")
        open(filepath, "w") do file
            println(file, velocity)
        end
    end
end

# __makefilemain__()

CUDA.device!(1)
__main__(SigmoidGradient, 10, 10, 10000)