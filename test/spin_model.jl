using Test, SurfaceProgrammableMaterial, DelimitedFiles, CUDA
using SurfaceProgrammableMaterial:spin_model_construction
using SurfaceProgrammableMaterial: unsafe_energy, unsafe_energy_temperature
using SurfaceProgrammableMaterial: update_temperature!


@testset "construct and energy" begin
    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(3, 2, CellularAutomata1D(110), false))
    @test model.nspin == 9
    states = spin_to_bool([-1, -1, -1, -1, -1, -1, 1, 1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([-1, -1, 1, -1, 1, -1, 1, -1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([-1, 1, -1, -1, 1, -1, 1, -1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([-1, 1, 1, -1, 1, -1, 1, -1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([1, -1, -1, -1, -1, -1, 1, 1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([1, -1, 1, -1, 1, -1, 1, -1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([1, 1, -1, -1, 1, -1, 1, -1, 1])
    @test energy(model, states) == -11
    states = spin_to_bool([1, 1, 1, -1, -1, -1, 1, -1, 1])
    @test energy(model, states) == -11

    for msk in 0:2^9-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:8])
        @test energy(model, states) >= -11
    end

    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(3, 3, CellularAutomata1D(110), true))
    states = spin_to_bool([-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
    @test energy(model, states) == -11 * 6
    states = spin_to_bool([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1])
    @test energy(model, states) == -11 * 6
    for msk in 0:2^(15)-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:14])
        @test energy(model, states) >= -11 * 6
    end
end

if CUDA.functional()
    @testset "energy" begin
        model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(12, 8, CellularAutomata1D(110), true))
        states = spin_to_bool(rand([-1, 1], model.nspin))
        energy_cpu = energy(model, reshape(states, :, 1))[1]
        states_gpu = CuArray(reshape(states, :, 1))
        energy_gpu = energy(model, states_gpu)[1]
        @show energy_cpu, energy_gpu
        @test energy_cpu == energy_gpu
    end
end

function unsafe_total_energy(model::IsingModel, states::AbstractMatrix)
    ret = zeros(Float64, size(states, 2))
    for inode in 1:model.nspin
        for ibatch in 1:size(states, 2)
            ret[ibatch] += unsafe_energy(states, model.interactions, model.onsites, 
                model.last_interaction_index, inode, ibatch)
            ret[ibatch] += states[inode, ibatch] ? model.onsites[inode] : -model.onsites[inode]    
        end
    end
    return ret ./ 2
end

function unsafe_total_energy_temperature(model::IsingModel, states::AbstractMatrix)
    ret = zeros(Float64, size(states, 2))
    for inode in 1:model.nspin
        for ibatch in 1:size(states, 2)
            ret[ibatch] += unsafe_energy_temperature(states, ones(Float64, size(states, 1), size(states, 2)), model.interactions, model.onsites, 
                model.last_interaction_index, inode, ibatch)
            ret[ibatch] += states[inode, ibatch] ? model.onsites[inode] : -model.onsites[inode]    
        end
    end
    return ret ./ 2
end

@testset "unsafe_energy" begin
    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(4, 2, CellularAutomata1D(110), false))
    for msk in 0:2^model.nspin-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:model.nspin-1])
        @test unsafe_total_energy(model, reshape(states, :, 1))[1] == energy(model, states)
    end
    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(3, 3, CellularAutomata1D(110), true))
    for msk in 0:2^model.nspin-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:model.nspin-1])
        @test unsafe_total_energy(model, reshape(states, :, 1))[1] == energy(model, states)
    end
end




@testset "unsafe_energy_temperature" begin
    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(4, 2, CellularAutomata1D(110), false))
    for msk in 0:2^model.nspin-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:model.nspin-1])
        @test unsafe_total_energy_temperature(model, reshape(states, :, 1))[1] == energy(model, states)
    end
    model = spin_model_construction(Float64, SimulatedAnnealingHamiltonian(3, 3, CellularAutomata1D(110), true))
    for msk in 0:2^model.nspin-1
        states = spin_to_bool([((msk >> i) & 1) == 1 ? 1 : -1 for i in 0:model.nspin-1])
        @test unsafe_total_energy_temperature(model, reshape(states, :, 1))[1] == energy(model, states)
    end
end

@testset "parallel_scheme_spin_model" begin
    for n in 4:12
        for m in 4:12
            sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
            eg = SigmoidGradient(40.0, 3.0, 1e-5)
            r = SpinSARuntime(Float64, sa, 1000)
            each_flip_group = parallel_scheme(r)
            @test all(sort!(vcat(each_flip_group...)) .== 1:nspin(r.model))
            for group in each_flip_group
                for spin in group
                    k = r.model.last_interaction_index[spin]
                    while k>0
                        interacted_spin = Int(r.model.interactions[k, 2])
                        @test (interacted_spin in group) == false
                        k = Int(r.model.interactions[k, 4])
                    end
                end
            end
        end
    end
end

@testset "simulated annealing" begin
   sa = SimulatedAnnealingHamiltonian(5, 5, CellularAutomata1D(110))
   nbatch = 1000
   annealing_time = 1050
   
   eg = SigmoidGradient(40.0, 3.0, 1e-5)
   r = SpinSARuntime(Float64, sa, nbatch)

   track_equilibration_pulse!(r, eg, annealing_time; flip_scheme = 1:nspin(r.model))
   state_energy = energy(r.model, r.state)
   success = count(x -> x == -11 * (sa.n * (sa.m-1)), state_energy)
   @show success
#    @show state_energy
end


function load_coupling(filename::String)
    data = readdlm(filename)
    is = Int.(view(data, :, 1)) .+ 1  #! @. means broadcast for the following functions, is here used correctly?
    js = Int.(view(data, :, 2)) .+ 1
    weights = data[:,3]
    num_spin = max(maximum(is), maximum(js))
    J = zeros(eltype(weights), num_spin, num_spin)
    @inbounds for (i, j, weight) = zip(is, js, weights)
        J[i,j] = weight
        J[j,i] = weight
    end
    return IsingModel(J)
end

@testset "simulated annealing via externel_std" begin
    # test 80 spins
    model = load_coupling(joinpath(@__DIR__, "externel_std/80_example.txt"))

    temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    r = SpinSARuntime(Float64, 30, model)
    track_equilibration_plane!(r, temp_scales, 100; transition_rule = Metropolis())
    state_energy = energy(r.model, r.state)
    @show state_energy, minimum(state_energy)
    @test minimum(state_energy) == -498

    # test 100 spins
    model = load_coupling(joinpath(@__DIR__, "externel_std/100_example.txt"))
    
    temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    r = SpinSARuntime(Float64, 30, model)
    track_equilibration_plane!(r, temp_scales, 50; transition_rule = Metropolis())
    state_energy = energy(r.model, r.state)
    @show state_energy, minimum(state_energy)
    @test minimum(state_energy) == -746

    # Please test following code on github CI
    # model = load_coupling(joinpath(@__DIR__, "externel_std/example.txt"))
    # @test model.nspin == 300
    # @test model.interactions[1, :] == [1, 2, 1, 0]
    # @test model.interactions[2, :] == [2, 1, 1, 0]
    # @test model.interactions[3, :] == [1, 3, 1, 1]
    # @test model.interactions[4, :] == [3, 1, 1, 0]
    
    # temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    # r = SpinSARuntime(Float64, 30, model)
    # track_equilibration_plane!(r, temp_scales, 2000; transition_rule = Metropolis())
    # state_energy = energy(r.model, r.state)
    # @show state_energy, minimum(state_energy)
    # @test minimum(state_energy) == -3858
end

if CUDA.functional()
    @testset "spin_model_pulse_equilibration_gpu" begin
        CUDA.device!(1)
        sa = SimulatedAnnealingHamiltonian(7, 5, CellularAutomata1D(110))
        nbatch = 1000
        annealing_time = 2050
    
        eg = SigmoidGradient(40.0, 3.0, 1e-5)
        r_cpu = SpinSARuntime(Float64, sa, nbatch)

        track_equilibration_pulse!(r_cpu, eg, annealing_time; flip_scheme = 1:nspin(r_cpu.model))
        state_energy = energy(r_cpu.model, r_cpu.state)
        success = count(x -> x == -11 * (sa.n * (sa.m-1)), state_energy)
        @show success

        r_gpu = CUDA.cu(SpinSARuntime(Float64, sa, nbatch))
        track_equilibration_pulse!(r_gpu, eg, annealing_time; flip_scheme = 1:nspin(r_gpu.model))
        state_energy_gpu = energy(r_cpu.model, Array(r_gpu.state))
        success_gpu = count(x -> x == -11 * (sa.n * (sa.m-1)), state_energy_gpu)
        @show success_gpu

        r_gpu_parallel = CUDA.cu(SpinSARuntime(Float64, sa, nbatch))
        track_equilibration_pulse!(r_gpu_parallel, eg, annealing_time; flip_scheme = CUDA.cu.(parallel_scheme(r_gpu_parallel)))
        state_energy_gpu_parallel = energy(r_cpu.model, Array(r_gpu_parallel.state))
        success_gpu_parallel = count(x -> x == -11 * (sa.n * (sa.m-1)), state_energy_gpu_parallel)
        @show success_gpu_parallel

        @show (success / nbatch, success_gpu / nbatch, success_gpu_parallel / nbatch)
        @test abs(success / nbatch - success_gpu / nbatch) <=0.05
        @test abs(success / nbatch - success_gpu_parallel / nbatch) <=0.05
    end
    
    # @testset "GPU vs std" begin
    #     # test 80 spins
    #     model = load_coupling("test/externel_std/80_example.txt")

    #     temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    #     r = CUDA.cu(SpinSARuntime(Float64, 30, model))
    #     track_equilibration_plane!(r, temp_scales, 100; transition_rule = Metropolis())
    #     state_energy = energy(model, Array(r.state))
    #     @show state_energy, minimum(state_energy)
    #     @test minimum(state_energy) == -498

    #     # test 100 spins
    #     model = load_coupling("test/externel_std/100_example.txt")
        
    #     temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    #     r = CUDA.cu(SpinSARuntime(Float64, 30, model))
    #     track_equilibration_plane!(r, temp_scales, 50; transition_rule = Metropolis())
    #     state_energy = energy(model, Array(r.state))
    #     @show state_energy, minimum(state_energy)
    #     @test minimum(state_energy) == -746

    #     # Please test following code on github CI
    #     # model = load_coupling("test/externel_std/example.txt")
    #     # @test model.nspin == 300
    #     # @test model.interactions[1, :] == [1, 2, 1, 0]
    #     # @test model.interactions[2, :] == [2, 1, 1, 0]
    #     # @test model.interactions[3, :] == [1, 3, 1, 1]
    #     # @test model.interactions[4, :] == [3, 1, 1, 0]
        
    #     # temp_scales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    #     # r = CUDA.cu(SpinSARuntime(Float64, 30, model))
    #     # track_equilibration_plane!(r, temp_scales, 2000; transition_rule = Metropolis())
    #     # state_energy = energy(r.model, Array(r.state))
    #     # @show state_energy, minimum(state_energy)
    #     # @test minimum(state_energy) == -3858
    # end
end
