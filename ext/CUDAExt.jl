module CUDAExt

using CUDA, SurfaceProgrammableMaterial

# convert the runtime to the CUDA version
function CUDA.cu(runtime::SARuntime)
    return SARuntime(runtime.hamiltonian, CuArray(runtime.state), CuArray(runtime.temperature))
end
function CUDA.cu(model::IsingModel)
    return IsingModel(model.nspin, CuArray(model.last_interaction_index), CuArray(model.interactions), CuArray(model.onsites))
end
function CUDA.cu(runtime::SpinSARuntime)
    return SpinSARuntime(runtime.hamiltonian, CUDA.cu(runtime.model), CuArray(runtime.state), CuArray(runtime.temperature))
end
SurfaceProgrammableMaterial._match_device(output::CuMatrix, offsets) = CuArray(offsets)
SurfaceProgrammableMaterial._flip_match_device(runtime::SARuntime{T, ET, <:CuMatrix{T}, <:CuMatrix{Bool}} where {T, ET}, spins) = length(spins) == 1 ? CuArray([spins]) : CuArray(spins)
SurfaceProgrammableMaterial._flip_match_device(runtime::SpinSARuntime{T, ET, <:CuMatrix{T}, <:CuMatrix{Bool}} where {T, ET}, spins) = length(spins) == 1 ? CuArray([spins]) : CuArray(spins)

function SurfaceProgrammableMaterial.step!(runtime::SARuntime{T, ET, <:CuMatrix{T}, <:CuMatrix{Bool}} where {T, ET},
            transition_rule::TransitionRule,
            simutanuous_flip_spins::CuVector
        )
    n = size(runtime.state, 2) * length(simutanuous_flip_spins)
    @inline function kernel(state::AbstractMatrix{Bool}, temperature::AbstractMatrix{T}, sa::SimulatedAnnealingHamiltonian{ET}, transition_rule::TransitionRule, n::Int, simutanuous_flip_spins::CuDeviceVector) where {T, ET<:CellularAutomata1D}
        idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        idx > n && return nothing
        ci = CartesianIndices((size(state, 2), length(simutanuous_flip_spins)))[idx]  # (ibatch, spin_id)
        SurfaceProgrammableMaterial.unsafe_step_kernel!(transition_rule, sa, state, temperature, ci.I[1], simutanuous_flip_spins[ci.I[2]])
        return nothing
    end

    # launch the kernel
    kernel = @cuda launch=false kernel(runtime.state, runtime.temperature, runtime.hamiltonian, transition_rule, n, simutanuous_flip_spins)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    CUDA.@sync kernel(runtime.state, runtime.temperature, runtime.hamiltonian, transition_rule, n, simutanuous_flip_spins; threads, blocks)
end

function SurfaceProgrammableMaterial.step!(runtime::SpinSARuntime{T, ET, <:CuMatrix{T}, <:CuMatrix{Bool}} where {T, ET},
            transition_rule::TransitionRule,
            simutanuous_flip_spins::CuVector
        )
    n = size(runtime.state, 2) * length(simutanuous_flip_spins)
    @inline function kernel(transition_rule::TransitionRule, temperature::AbstractMatrix{T},
    state::AbstractMatrix{Bool}, interactions::AbstractMatrix, onsites::AbstractArray, last_interaction_index::AbstractArray,
    simutanuous_flip_spins::CuDeviceVector) where T
        idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        idx > n && return nothing
        ci = CartesianIndices((size(state, 2), length(simutanuous_flip_spins)))[idx]  # (ibatch, spin_id)
        SurfaceProgrammableMaterial.unsafe_step_kernel!(transition_rule, temperature, state, interactions, onsites, last_interaction_index, ci.I[1], simutanuous_flip_spins[ci.I[2]])
        return nothing
    end

    # launch the kernel
    kernel = @cuda launch=false kernel(transition_rule, runtime.temperature, runtime.state, runtime.model.interactions, runtime.model.onsites, runtime.model.last_interaction_index, simutanuous_flip_spins)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    CUDA.@sync kernel(transition_rule, runtime.temperature, runtime.state, runtime.model.interactions, runtime.model.onsites, runtime.model.last_interaction_index, simutanuous_flip_spins; threads, blocks)
end

function energy_kernel!(interactions::AbstractMatrix, state::AbstractMatrix, energy::AbstractArray, ibatch, iinter)
    sx, sy = state[Int(interactions[iinter, 1]), ibatch], state[Int(interactions[iinter, 2]), ibatch]
    energy[ibatch] += sx ⊻ sy ? -interactions[iinter, 3] : interactions[iinter, 3]
end

function SurfaceProgrammableMaterial.energy(model::IsingModel, state::CuMatrix{Bool})
    ret = CuArray(zeros(size(state, 2)))
    interactions = CuArray(model.interactions)
    onsites = CuArray(model.onsites)
    n = size(state, 2)
    @inline function kernel(interactions::AbstractMatrix, onsites::AbstractArray, state::AbstractMatrix, energy::AbstractArray, nspin::Integer)
        idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        idx > n && return nothing
        for iinter in 1:2:size(interactions, 1)
            energy_kernel!(interactions, state, energy, idx, iinter)
        end
        for ispin in 1:nspin
            energy[idx] += state[ispin, idx] ? onsites[ispin] : -onsites[ispin]
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(interactions, onsites, state, ret, model.nspin)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    CUDA.@sync kernel(interactions, onsites, state, ret, model.nspin; threads, blocks)
    
    return Array(ret)
end

end