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
    return (2*SurfaceProgrammableMaterial.cutoff_distance(tg) + m) / (final_sweep_time - 1)
end

# epsilon determines the lowest_temperature
# Tmax denote the highest_temperature
function __main__(typetemp::Type, n::Int, m::Int, nbatch::Int; Tmax=2.0, epsilon=1e-5, width=1.0, input_fixed = true)
    lowest_temperature = -1/log(epsilon)
    tg = typetemp(Tmax, width, lowest_temperature)
    flip_group = parallel_scheme(SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110)); input_fixed)
    input_fixed == true && @assert sort!(vcat(flip_group...)) == collect(n+1:n*m) "flip_group is not correct"
    moving_speed = evaluate_p_percentage_velocity(tg, n, m, nbatch, 0.50; 
                    CUDA_functional = true, flip_scheme = CUDA.cu.(flip_group))
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
    input_fixed = parse(Bool, ARGS[8])
    cuda_device = parse(Int, ARGS[9])

    CUDA.device!(cuda_device)
    for m in m_minimum:m_step:m_maximum
        velocity = __main__(temperature_type, n, m, nbatch; width, input_fixed)
        filepath = joinpath(@__DIR__, "data_$(temperature_type)/n=$(n)_m=$(m)_width=$(width)_nbatch=$(nbatch).txt")
        open(filepath, "w") do file
            println(file, velocity)
        end
    end
end

__makefilemain__()

# CUDA.device!(1)
# __main__(SigmoidGradient, 20, 20, 10000)