using Test, CUDA
using SurfaceProgrammableMaterial
using SurfaceProgrammableMaterial: natom, atoms, hasparent
using SurfaceProgrammableMaterial: evaluate_parent, calculate_energy
using SurfaceProgrammableMaterial: parent_nodes, child_nodes
using SurfaceProgrammableMaterial: SimulatedAnnealingHamiltonian
using SurfaceProgrammableMaterial: get_parallel_flip_id
using SurfaceProgrammableMaterial: random_state

@testset "basic_hamiltonian" begin
    sa = SimulatedAnnealingHamiltonian(2, 3)
    @test natom(sa) == 6
    @test atoms(sa) == [1, 2, 3, 4, 5, 6]
    @test hasparent(sa, 1) == false
    @test hasparent(sa, 3) == true
end

@testset "energy_calculation" begin
    # 0001
    # 1110
    # 0110
    # 1011
    sa = SimulatedAnnealingHamiltonian(4, 4)
    state = reshape([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1], 16, 1)
    energy_gradient = [1]
    ibatch = 1
    @test length(state) == natom(sa)
    @test evaluate_parent(sa, state, energy_gradient, 5, ibatch) == 1
    @test evaluate_parent(sa, state, energy_gradient, 6, ibatch) == 0
    @test evaluate_parent(sa, state, energy_gradient, 7, ibatch) == 0
    @test evaluate_parent(sa, state, energy_gradient, 8, ibatch) == 0
    @test evaluate_parent(sa, state, energy_gradient, 10, ibatch) == 0
    @test calculate_energy(sa, state, energy_gradient, ibatch) == 3
end

@testset "parent_nodes_child_nodes" begin
    sa = SimulatedAnnealingHamiltonian(5, 4)
    @test parent_nodes(sa, 6) == (5, 1, 2)
    @test parent_nodes(sa, 7) == (1, 2, 3)
    @test parent_nodes(sa, 10) == (4, 5, 1)
    @test parent_nodes(sa, 11) == (10, 6, 7)
    
    @test child_nodes(sa, 1) == (10, 6, 7)
    @test child_nodes(sa, 3) == (7, 8, 9)
    @test child_nodes(sa, 5) == (9, 10, 6)
    @test child_nodes(sa, 6) == (15, 11, 12)
end

function test_flip_id(sa::SimulatedAnnealingHamiltonian)
    each_flip_group = get_parallel_flip_id(sa)
    @test sum(length.(each_flip_group)) == sa.n*sa.m

    total_flip_id = sort(vcat(each_flip_group...))
    @test total_flip_id == 1:sa.n*sa.m

    for this_flip_group in each_flip_group
        for i in 1:length(this_flip_group)
            for j in i+1:length(this_flip_group)
                related_gadgets_i = []
                related_gadgets_j = []
                if this_flip_group[i] > sa.n
                    related_gadgets_i = vcat(related_gadgets_i, [this_flip_group[i]])
                end
                if this_flip_group[j] > sa.n
                    related_gadgets_j = vcat(related_gadgets_j, [this_flip_group[j]])
                end
                if this_flip_group[i] <= sa.n*(sa.m-1)
                    related_gadgets_i = vcat(related_gadgets_i, child_nodes(sa, this_flip_group[i]))
                end
                if this_flip_group[j] <= sa.n*(sa.m-1)
                    related_gadgets_j = vcat(related_gadgets_j, child_nodes(sa, this_flip_group[j]))
                end
                @test length(intersect(related_gadgets_i, related_gadgets_j)) == 0
            end
        end
    end
end

@testset "parallel_flip_id" begin
    for n in 3:12
        for m in 3:12
            sa = SimulatedAnnealingHamiltonian(n, m)
            test_flip_id(sa)
        end
    end
end

@testset "pulse_equilibration_cpu" begin
    sa = SimulatedAnnealingHamiltonian(8, 8)
    nbatch = 2000
    state = random_state(sa, nbatch)
    pulse_gradient = 1.3
    pulse_amplitude = 10.0
    pulse_width = 1.0
    annealing_time = 2000

    track_equilibration_pulse_cpu!(HeatBath(), Exponentialtype(),
     sa, state, pulse_gradient, pulse_amplitude, pulse_width, annealing_time;
      accelerate_flip = true)
    
    state_energy = [calculate_energy(sa, state, fill(1.0, nbatch), i) for i in 1:nbatch]
    success = count(x -> x == 0, state_energy)
    @test abs((success / nbatch) - 0.55) <= 0.1

    sequential_flip_state = random_state(sa, nbatch)
    track_equilibration_pulse_cpu!(HeatBath(), Exponentialtype(),
     sa, sequential_flip_state, pulse_gradient, pulse_amplitude, pulse_width, annealing_time;
      accelerate_flip = false)
    sequential_flip_state_energy = [calculate_energy(sa, sequential_flip_state, fill(1.0, nbatch), i) for i in 1:nbatch]
    sequential_flip_success = count(x -> x==0, sequential_flip_state_energy)
    @info success, sequential_flip_success
    @test abs((sequential_flip_success - success) / nbatch) <= 0.05
end

if CUDA.functional()
    @testset "pulse_equilibration_gpu" begin
        sa = SimulatedAnnealingHamiltonian(8, 8)
        nbatch = 5000
        cpu_state = random_state(sa, nbatch)
        gpu_state = CuArray(random_state(sa, nbatch))
        pulse_gradient = 1.3
        pulse_amplitude = 10.0
        pulse_width = 1.0
        annealing_time = 2000

        track_equilibration_pulse_cpu!(HeatBath(), Exponentialtype(),
        sa, cpu_state, pulse_gradient, pulse_amplitude, pulse_width, annealing_time;
        accelerate_flip = true)

        track_equilibration_pulse_gpu!(HeatBath(), Exponentialtype(),
        sa, gpu_state, pulse_gradient, pulse_amplitude, pulse_width, annealing_time;
        accelerate_flip = true)   

        cpu_state_energy = [calculate_energy(sa, cpu_state, fill(1.0, nbatch), i) for i in 1:nbatch]
        gpu_state_energy = [calculate_energy(sa, Array(gpu_state), fill(1.0, nbatch), i) for i in 1:nbatch]
        cpu_success = count(x -> x == 0, cpu_state_energy)
        gpu_success = count(x -> x == 0, gpu_state_energy)
        @info cpu_success, gpu_success
        @test abs((cpu_success - gpu_success) / nbatch) <= 0.03
    end
end