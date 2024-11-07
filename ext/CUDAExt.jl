module CUDAExt

using CUDA, SurfaceProgrammableMaterial

# convert the runtime to the CUDA version
function CUDA.cu(runtime::SARuntime)
    return SARuntime(runtime.hamiltonian, CuArray(runtime.state), CuArray(runtime.temperature))
end
SurfaceProgrammableMaterial._match_device(output::CuMatrix, offsets) = CuArray(offsets)

function SurfaceProgrammableMaterial.step!(runtime::SARuntime{T, ET, <:CuMatrix{T}, <:CuMatrix{Bool}} where {T, ET},
            transition_rule::TransitionRule,
            simutanuous_flip_spins
        )

    n = size(runtime.state, 2) * length(simutanuous_flip_spins)
    @inline function kernel(state::AbstractMatrix{Bool}, temperature::AbstractMatrix{T}, sa::SimulatedAnnealingHamiltonian{ET}, transition_rule::TransitionRule, n::Int) where {T, ET<:CellularAutomata1D}
        idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        idx > n && return nothing
        ci = CartesianIndices(state)[idx]  # (ibatch, spin)
        SurfaceProgrammableMaterial.unsafe_step_kernel!(transition_rule, sa, state, temperature, ci.I[2], ci.I[1])
        return nothing
    end

    # launch the kernel
    kernel = @cuda launch=false kernel(runtime.state, runtime.temperature, runtime.hamiltonian, transition_rule, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    CUDA.@sync kernel(runtime.state, runtime.temperature, runtime.hamiltonian, transition_rule, n; threads, blocks)
end

end
