using Test, SurfaceProgrammableMaterial
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

function unsafe_total_energy(model::IsingModel, states::AbstractArray)
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

function unsafe_total_energy_temperature(model::IsingModel, states::AbstractArray)
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