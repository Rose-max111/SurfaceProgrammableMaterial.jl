using SurfaceProgrammableMaterial
using SurfaceProgrammableMaterial: random_state, calculate_energy
using SurfaceProgrammableMaterial: track_equilibration_pulse_gpu!, track_equilibration_pulse_reverse_cpu!
using CUDA
using SurfaceProgrammableMaterial: get_parallel_flip_id
using CairoMakie
using SurfaceProgrammableMaterial: temp_calculate

function testplot(xdata, ydata)
    fig = Figure()
    ax = Axis(fig[1, 1])
    scatter!(ax, xdata, ydata)
    fig
end

function trueoutput(width, depth, inputs) # inputs is the first layer calculate depth layer
    statearray = zeros(Bool, (width, 1))
    next_array = zeros(Bool, (width, 1))
    for i in 1:width
        statearray[i] = inputs[i]
    end
    # @info statearray
    for i in 1:depth-1
        for mid in 1:width
            pre = mod1(mid-1, width)
            suf = mod1(mid+1, width)
            next_array[mid] = rule110(statearray[pre], statearray[mid], statearray[suf])
        end
        # @info next_array
        statearray = copy(next_array)
    end
    return statearray
end


function trans_vaild_output!(state, sa, nbatch)
    for ibatch in 1:nbatch
        this_input = copy(state[1:sa.n, ibatch])
        true_output = trueoutput(sa.n, sa.m, this_input)
        state[sa.n*sa.m-sa.n+1:sa.n*sa.m, ibatch] .= true_output
    end
end

function evaluate_50percent_time_cpu(temprule::TempcomputeRule, width::Integer, depth::Integer, gauss_width, energy_gradient)
    sa = SimulatedAnnealingHamiltonian(width, depth)
    nbatch = 5000

    anneal_time = 0
    max_try = 2
    while max_try > 0
        next_try = anneal_time + max_try
        
        state = random_state(sa, nbatch)
        @info "Stage1, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_cpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        state_energy = [calculate_energy(sa, state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 1, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            break
        else
            max_try *= 2
        end
    end
    anneal_time += max_try / 2
    max_try /= 2
    if max_try <= 1
        return anneal_time
    end
    while max_try != 1
        next_try = anneal_time + max_try
        
        state = random_state(sa, nbatch)
        @info "Stage2, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_cpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        state_energy = [calculate_energy(sa, state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        # @info "Stage 2, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            max_try /=2
            continue
        else
            max_try /=2
            anneal_time = next_try
        end
    end
    return anneal_time
end


function evaluate_50percent_time_gpu(temprule::TempcomputeRule, width::Integer, depth::Integer, gauss_width, energy_gradient)
    sa = SimulatedAnnealingHamiltonian(width, depth)
    nbatch = 10000

    anneal_time = 0.0
    max_try = 2.0
    while max_try > 0
        next_try = anneal_time + max_try
        
        state = CuArray(random_state(sa, nbatch))
        @info "Stage1, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_gpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 1, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            break
        else
            max_try *= 2
        end
    end
    anneal_time += max_try / 2
    max_try /= 4
    if max_try <= 1
        return anneal_time + 1
    end
    while max_try > 32
        next_try = anneal_time + max_try
        
        state = CuArray(random_state(sa, nbatch))
        @info "Stage2, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_gpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 2, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            max_try /=2
            continue
        else
            max_try /=2
            anneal_time = next_try
        end
    end
    return anneal_time + 1
end

function evaluate_50percent_time_reverse_gpu(temprule::TempcomputeRule, width::Integer, depth::Integer, gauss_width, energy_gradient)
    sa = SimulatedAnnealingHamiltonian(width, depth)
    nbatch = 5000

    anneal_time = 0.0
    max_try = 2.0
    while max_try > 0
        next_try = anneal_time + max_try
        
        state = CuArray(random_state(sa, nbatch))
        @info "Stage1, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_reverse_gpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 1, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            break
        else
            max_try *= 2
        end
    end
    anneal_time += max_try / 2
    max_try /= 4
    if max_try <= 1
        return anneal_time + 1
    end
    while max_try > 32
        next_try = anneal_time + max_try
        
        state = CuArray(random_state(sa, nbatch))
        @info "Stage2, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_pulse_reverse_gpu!(HeatBath(), temprule, sa, state, energy_gradient, 10.0, gauss_width, next_try; accelerate_flip = true)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 2, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            max_try /=2
            continue
        else
            max_try /=2
            anneal_time = next_try
        end
    end
    return anneal_time + 1
end

function evaluate_50percent_time_fixlayer_gpu(width::Integer, depth::Integer, fixedinput)
    sa = SimulatedAnnealingHamiltonian(width, depth)
    nbatch = 5000

    anneal_time = 0.0
    max_try = 2.0
    while max_try > 0
        next_try = anneal_time + max_try
        
        pre_state = random_state(sa, nbatch)
        if fixedinput == false
             trans_vaild_output!(pre_state, sa, nbatch)
        end
        state = CuArray(pre_state)
        @info "Stage1, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_fixedlayer_gpu!(HeatBath(), sa, state, Int(next_try); accelerate_flip = true, fixedinput = fixedinput)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 1, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            break
        else
            max_try *= 2
        end
    end
    anneal_time += max_try / 2
    max_try /= 4
    if max_try <= 1
        return anneal_time + 1
    end
    while max_try > 32
        next_try = anneal_time + max_try
        
        pre_state = random_state(sa, nbatch)
        if fixedinput == false
             trans_vaild_output!(pre_state, sa, nbatch)
        end
        state = CuArray(pre_state)
        @info "Stage2, max_try = $max_try, next_try = $next_try, begin annealing"
        # @info "$Temp_sa"
        @time track_equilibration_fixedlayer_gpu!(HeatBath(), sa, state, Int(next_try); accelerate_flip = true, fixedinput = fixedinput)
        @info "finish annealing"

        cpu_state = Array(state)
        state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        success = count(x -> x == 0, state_energy)
        @info "Stage 2, now max_try = $max_try, success time = $success, anneal_time = $anneal_time"
        if 1.0 * success / nbatch >= 0.49
            max_try /=2
            continue
        else
            max_try /=2
            anneal_time = next_try
        end
    end
    return anneal_time + 1
end


# evaluate_50percent_time_cpu(12, 8, 1.0, 1.5)

width = ARGS[1]
depth = ARGS[2]
gauss_width = ARGS[3]
λ = ARGS[4]
device = ARGS[5]

width = parse(Int, width)
depth = parse(Int, depth)
gauss_width = parse(Float64, gauss_width)
λ = parse(Float64, λ)
device = parse(Int, device)

# # @info "this time try width = $width, depth = $depth, λ = $λ, gauss_width = $(gauss_width)"
CUDA.device!(device)
# evaluate_time = evaluate_50percent_time_reverse_gpu(Exponentialtype(), width, depth, gauss_width, λ)
evaluate_time = evaluate_50percent_time_fixlayer_gpu(width, depth, false)

# # # @info "width = $width, depth = $depth, λ = $λ, evaluate_time = $evaluate_time"

filepath = joinpath(@__DIR__, "data_toymodel_fixoutput/W=$(width)_D=$(depth)_GW=$(gauss_width)_E=$(λ).txt")
open(filepath,"w") do file
    println(file, evaluate_time)
end
