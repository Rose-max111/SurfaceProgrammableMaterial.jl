module CUDAExt

using CUDA
import SurfaceProgrammableMaterial: unsafe_step_kernel!, parallel_scheme, nspin, SimulatedAnnealingHamiltonian, TransitionRule, TemperatureGradient

function step!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::CuMatrix, energy_gradient::AbstractArray, Temp, node=nothing)
    @inline function kernel(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, node=nothing)
        ibatch = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        if ibatch <= size(state, 2)
            unsafe_step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, node)
        end
        return nothing
    end
    kernel = @cuda launch=false kernel(rule, sa, state, energy_gradient, Temp, node)
    config = launch_configuration(kernel.fun)
    threads = min(size(state, 2), config.threads)
    blocks = cld(size(state, 2), threads)
    CUDA.@sync kernel(rule, sa, state, energy_gradient, Temp, node; threads, blocks)
    state
end

function step_parallel!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::CuMatrix, energy_gradient::AbstractArray, Temp, flip_id)
    @inline function kernel(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, energy_gradient::AbstractArray, Temp, flip_id)
        id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        Nx = size(state, 2)
        Ny = length(flip_id)
        cind = CartesianIndices((Nx, Ny))
        for k in id:stride:Nx*Ny
            ibatch = cind[k][1]
            id = cind[k][2]
            unsafe_step_kernel!(rule, sa, state, energy_gradient, Temp, ibatch, flip_id[id])
        end
        return nothing
    end
    kernel = @cuda launch=false kernel(rule, sa, state, energy_gradient, Temp, flip_id)
    config = launch_configuration(kernel.fun)
    threads = min(size(state, 2) * length(flip_id), config.threads)
    blocks = cld(size(state, 2) * length(flip_id), threads)
    CUDA.@sync kernel(rule, sa, state, energy_gradient, Temp, flip_id; threads, blocks)
    state
end

function track_equilibration_collective_temperature!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::CuMatrix,
                                        temperature,
                                        annealing_time; accelerate_flip = false)
    for t in 1:annealing_time
        singlebatch_temp = fill(Float32(temperature), sa.m-1)
        Temp = CuArray(fill(singlebatch_temp, size(state, 2)))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, thisatom)
            end
        else
            flip_list = parallel_scheme(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, CuArray(eachflip))
            end
        end
    end
end

function track_equilibration_pulse_reverse!(rule::TransitionRule,
                                        temprule::TemperatureGradient,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::CuMatrix, 
                                        pulse_gradient, 
                                        pulse_amplitude,
                                        pulse_width,
                                        annealing_time; accelerate_flip = false
                                        )    
    midposition = midposition_calculate(temprule, pulse_amplitude, pulse_width, pulse_gradient)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    midposition = sa.m - 1.0 + 1.0 - midposition
    @info "each_movement = $each_movement"

    for t in 1:annealing_time
        singlebatch_temp = toymodel_pulse(temprule, sa, pulse_amplitude, pulse_width, midposition, pulse_gradient)
        Temp = CuArray(fill(Float32.(singlebatch_temp), size(state, 2)))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, CuArray(fill(1.0, size(state, 2))), Temp, thisatom)
            end
        else
            flip_list = parallel_scheme(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, CuArray(fill(1.0, size(state, 2))), Temp, CuArray(eachflip))
            end
        end
        midposition -= each_movement
    end
    return sa
end

function track_equilibration_fixedlayer!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::CuMatrix,  
                                        annealing_time; accelerate_flip = false, fixedinput = false)
    Temp_list = reverse(range(1e-5, 10.0, annealing_time))
    for this_temp in Temp_list
        # @info "yes"
        singlebatch_temp = Tuple(fill(Float32(this_temp), sa.m-1))
        # @info "$(typeof(singlebatch_temp))"
        Temp = CuArray((fill(singlebatch_temp, size(state, 2))))
        if accelerate_flip == false
            if fixedinput == false # this time fix output
                for thisatom in 1:nspin(sa)-sa.n
                    step!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, thisatom)
                end
            else # this time fix input
                for thisatom in sa.n+1:nspin(sa)
                    step!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, thisatom)
                end
            end
        else
            flip_list = parallel_scheme(SimulatedAnnealingHamiltonian(sa.n, sa.m-1)) # original fix output
            if fixedinput == true # this time fix input
                flip_list = [[x + sa.n for x in inner_vec] for inner_vec in flip_list]
            end
            for eachflip in flip_list
                step_parallel!(rule, sa, state, CuArray(fill(1.0f0, size(state, 2))), Temp, CuArray(eachflip))
            end
        end
    end
end

function track_equilibration_pulse!(rule::TransitionRule,
                                        temprule::TemperatureGradient,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::CuMatrix, 
                                        pulse_gradient, 
                                        pulse_amplitude,
                                        pulse_width,
                                        annealing_time; accelerate_flip = false
                                        )    
    midposition = midposition_calculate(temprule, pulse_amplitude, pulse_width, pulse_gradient)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    @info "each_movement = $each_movement"

    for t in 1:annealing_time
        singlebatch_temp = toymodel_pulse(temprule, sa, pulse_amplitude, pulse_width, midposition, pulse_gradient)
        temperature = CuArray(fill(Float32.(singlebatch_temp), size(state, 2)))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, CuArray(fill(1.0, size(state, 2))), temperature, thisatom)
            end
        else
            flip_list = parallel_scheme(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, CuArray(fill(1.0, size(state, 2))), temperature, CuArray(eachflip))
            end
        end
        midposition += each_movement
    end
    return sa
end


end
