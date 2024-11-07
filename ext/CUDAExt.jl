module CUDAExt

using CUDA, SurfaceProgrammableMaterial
import SurfaceProgrammableMaterial: parallel_scheme, nspin, SimulatedAnnealingHamiltonian, TransitionRule, TemperatureGradient
import SurfaceProgrammableMaterial: cutoff_distance, temperature_matrix!
import SurfaceProgrammableMaterial: unsafe_parent_nodes, unsafe_child_nodes
import SurfaceProgrammableMaterial: update_temperature!
import SurfaceProgrammableMaterial: prob_accept


function SurfaceProgrammableMaterial.update_temperature!(runtime::SARuntime{T, ET, <:CuMatrix, <:CuMatrix{Bool}} where {T, ET}, temprule::ColumnWiseGradient, t::Integer, annealing_time::Integer, reverse_direction::Bool)
    # @info "GPU update_temperature!"
    dcut = cutoff_distance(temprule)
    sa = runtime.hamiltonian
    each_movement = (dcut * 2 + sa.m) / (annealing_time - 1)
    middle_position = reverse_direction ? sa.m + dcut - t * each_movement : -dcut + t * each_movement
    
    temporary_temperature = ones(Float32, (sa.n, sa.m))
    temperature_matrix!(temporary_temperature, temprule, sa, middle_position)
    temporary_temperature = CuArray(reshape(temporary_temperature, :))

    @inline function kernel(temperature, temporary_temperature)
        id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        Nx = size(temperature, 2)
        Ny = size(temperature, 1)
        cind = CartesianIndices((Nx, Ny))
        for k in id:stride:Nx*Ny
            ibatch = cind[k][1]
            spin = cind[k][2]
            temperature[spin, ibatch] = temporary_temperature[spin]
        end
        return nothing
    end
    # @info typeof(temporary_temperature)
    kernel = @cuda launch=false kernel(runtime.temperature, temporary_temperature)
    config = launch_configuration(kernel.fun)
    threads = min(size(runtime.temperature, 2) * size(runtime.temperature, 1), config.threads)
    blocks = cld(size(runtime.temperature, 2) * size(runtime.temperature, 1), threads)
    CUDA.@sync kernel(runtime.temperature, temporary_temperature; threads, blocks)
end

function SurfaceProgrammableMaterial.SARuntime_CUDA(::Type{T}, sa::SimulatedAnnealingHamiltonian{ET}, nbatch::Integer) where {T, ET}
    state = CuArray(random_state(sa, nbatch))
    temperature = CuArray(ones(T, nspin(sa), nbatch))
    @assert typeof(state) <: AbstractMatrix{Bool} " state must be a matrix of Bool"
    # @info "$(typeof(state)), $(typeof(state) <: AbstractMatrix{Bool})"
    return SARuntime(sa, state, temperature)
end

@inline function unsafe_energy_temperature(sa::SimulatedAnnealingHamiltonian,
    state::AbstractMatrix,
    temperature::AbstractMatrix,
    inode::Integer, ibatch::Integer)
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        # NOTE: two sqrt is much faster than x^0.25
        Teff = sqrt(sqrt(temperature[a, ibatch] * temperature[b, ibatch] * temperature[c, ibatch] * temperature[inode, ibatch]))
        return trueoutput ⊻ state[inode, ibatch], Teff
    end
end

@inline function unsafe_step_kernel!(transition_rule::TransitionRule, 
    sa::SimulatedAnnealingHamiltonian,
    state::AbstractMatrix, 
    temperature::AbstractMatrix, 
    ibatch::Integer, 
    node:: Integer)
    # @info "begin kernel"
    @inbounds begin
        grid = CartesianIndices((sa.n, sa.m))
        i, j = grid[node].I  # proposed position

        ΔE_over_T_previous_layer = j > 1 ? ((ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch); (one(Float32) - 2 * ΔE) / Teff) : zero(Float32)
        ΔE_over_T_next_layer = zero(Float32)
        if j < sa.m # not the last layer
            # calculate the energy from the child nodes
            cnodes = unsafe_child_nodes(sa, node)
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer -= ΔE / Teff
            end
            # calculate the energy from the child nodes after flipping
            state[node, ibatch] ⊻= true
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer += ΔE / Teff
            end
            state[node, ibatch] ⊻= true
        end
        ΔE_over_T = ΔE_over_T_previous_layer + ΔE_over_T_next_layer
        if rand() < prob_accept(transition_rule, ΔE_over_T)
            state[node, ibatch] ⊻= true
        end
    end
end

function step!(runtime::SARuntime{T, ET, <:CuMatrix, <:CuMatrix{Bool}} where {T, ET},
            transition_rule::TransitionRule,
            simutanuous_flip_spins::CuVector
            )
    @inline function kernel(
        transition_rule::TransitionRule,
        sa::SimulatedAnnealingHamiltonian, 
        state::CuDeviceMatrix, 
        temperature::CuDeviceMatrix,
        simutanuous_flip_spins::CuDeviceVector)
        id = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
        Nx = size(state, 2)
        Ny = length(simutanuous_flip_spins)
        cind = CartesianIndices((Nx, Ny))
        for k in id:stride:Nx*Ny
            ibatch = cind[k][1]
            spin_id = cind[k][2]
            unsafe_step_kernel!(transition_rule, sa, state, temperature, ibatch, simutanuous_flip_spins[spin_id])
        end
        return nothing
    end
 
    kernel = @cuda launch=false kernel(
        transition_rule,
        runtime.hamiltonian, runtime.state, runtime.temperature, simutanuous_flip_spins)
    config = launch_configuration(kernel.fun)
    threads = min(size(runtime.state, 2) * length(simutanuous_flip_spins), config.threads)
    blocks = cld(size(runtime.state, 2) * length(simutanuous_flip_spins), threads)
    CUDA.@sync kernel(
        transition_rule, 
        runtime.hamiltonian, runtime.state, runtime.temperature, simutanuous_flip_spins; threads, blocks)
end

function SurfaceProgrammableMaterial.track_equilibration_pulse!(
            runtime::SARuntime{T, ET, <:CuMatrix, <:CuMatrix{Bool}} where {T, ET},
            temprule::TemperatureGradient,
            annealing_time;
            tracker = nothing,
            flip_scheme = 1:nspin(runtime.hamiltonian),
            transition_rule::TransitionRule = HeatBath(),
            reverse_direction::Bool=false
        ) 
    @info "GPU track_equilibration_pulse!"

    if flip_scheme == 1:nspin(runtime.hamiltonian)
        flip_scheme = parallel_scheme(runtime.hamiltonian)
    end
    update_temperature!(runtime, temprule, 0, annealing_time, reverse_direction)
    tracker !== nothing && track!(tracker, copy(runtime.state), copy(runtime.temperature))
    for t in 1:annealing_time
        update_temperature!(runtime, temprule, t, annealing_time, reverse_direction)
        for spins in flip_scheme
            step!(runtime, transition_rule, length(spins) == 1 ? CuArray([spins]) : CuArray(spins))
        end
        tracker !== nothing && track!(tracker, copy(runtime.state), copy(runtime.temperature))
    end
end

end
