using Test, CUDA
using SurfaceProgrammableMaterial
using SurfaceProgrammableMaterial: nspin, spins
using SurfaceProgrammableMaterial: unsafe_evaluate_parent
using SurfaceProgrammableMaterial: unsafe_parent_nodes, unsafe_child_nodes
using SurfaceProgrammableMaterial: SimulatedAnnealingHamiltonian
using SurfaceProgrammableMaterial: get_parallel_flip_id
using SurfaceProgrammableMaterial: random_state

@testset "basic_hamiltonian" begin
    sa = SimulatedAnnealingHamiltonian(2, 3, CellularAutomata1D(110))
    @test nspin(sa) == 6
    @test spins(sa) == [1, 2, 3, 4, 5, 6]
end

@testset "temperature gradient" begin
    eg = ExponentialGradient(1.0, 1.0, 1e-5)
    d = SurfaceProgrammableMaterial.cutoff_distance(eg)
    # one 1e-5 is the lowest temperature, another 1e-5 is from the wave packet
    @test SurfaceProgrammableMaterial.evaluate_temperature(eg, d) ≈ 2e-5
    gg = GaussianGradient(1.0, 1.0, 1e-5)
    d = SurfaceProgrammableMaterial.cutoff_distance(gg)
    @test SurfaceProgrammableMaterial.evaluate_temperature(gg, d) ≈ 2e-5
end

@testset "energy_calculation" begin
    # 0001
    # 1110
    # 0110
    # 1011
    sa = SimulatedAnnealingHamiltonian(4, 4, CellularAutomata1D(110))
    state = reshape([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1], 16, 1)
    ibatch = 1
    @test length(state) == nspin(sa)
    @test unsafe_evaluate_parent(sa, state, 5, ibatch) == 1
    @test unsafe_evaluate_parent(sa, state, 6, ibatch) == 0
    @test unsafe_evaluate_parent(sa, state, 7, ibatch) == 0
    @test unsafe_evaluate_parent(sa, state, 8, ibatch) == 0
    @test unsafe_evaluate_parent(sa, state, 10, ibatch) == 0
    @test energy(sa, state)[ibatch] == 3
end

@testset "parent_nodes_child_nodes" begin
    sa = SimulatedAnnealingHamiltonian(5, 4, CellularAutomata1D(110))
    @test unsafe_parent_nodes(sa, 21) == (20, 16, 17)
    @test unsafe_parent_nodes(sa, 6) == (5, 1, 2)
    @test unsafe_parent_nodes(sa, 7) == (1, 2, 3)
    @test unsafe_parent_nodes(sa, 10) == (4, 5, 1)
    @test unsafe_parent_nodes(sa, 11) == (10, 6, 7)
    
    @test unsafe_child_nodes(sa, 1) == (10, 6, 7)
    @test unsafe_child_nodes(sa, 3) == (7, 8, 9)
    @test unsafe_child_nodes(sa, 5) == (9, 10, 6)
    @test unsafe_child_nodes(sa, 6) == (15, 11, 12)
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
                    related_gadgets_i = vcat(related_gadgets_i, unsafe_child_nodes(sa, this_flip_group[i]))
                end
                if this_flip_group[j] <= sa.n*(sa.m-1)
                    related_gadgets_j = vcat(related_gadgets_j, unsafe_child_nodes(sa, this_flip_group[j]))
                end
                @test length(intersect(related_gadgets_i, related_gadgets_j)) == 0
            end
        end
    end
end

@testset "parallel_flip_id" begin
    for n in 3:12
        for m in 3:12
            sa = SimulatedAnnealingHamiltonian(n, m, CellularAutomata1D(110))
            test_flip_id(sa)
        end
:wait    end
end

@testset "pulse_equilibration_cpu" begin
    sa = SimulatedAnnealingHamiltonian(8, 8, CellularAutomata1D(110))
    nbatch = 2000
    state = random_state(sa, nbatch)
    pulse_amplitude = 10.0
    pulse_width = 1/1.3
    annealing_time = 2000

    eg = ExponentialGradient(pulse_amplitude, pulse_width, 1e-5)
    track_equilibration_pulse_cpu!(HeatBath(), eg, sa, state, annealing_time; accelerate_flip = true)
    
    state_energy = energy(sa, state)
    success = count(x -> x == 0, state_energy)
    @test abs((success / nbatch) - 0.55) <= 0.1

    sequential_flip_state = random_state(sa, nbatch)
    track_equilibration_pulse_cpu!(HeatBath(), eg, sa, sequential_flip_state, annealing_time; accelerate_flip = false)
    sequential_flip_state_energy = energy(sa, sequential_flip_state)
    sequential_flip_success = count(x -> x==0, sequential_flip_state_energy)
    @info success, sequential_flip_success
    @test abs((sequential_flip_success - success) / nbatch) <= 0.05
end

if CUDA.functional()
    @testset "pulse_equilibration_gpu" begin
        sa = SimulatedAnnealingHamiltonian(8, 8, CellularAutomata1D(110))
        nbatch = 5000
        cpu_state = random_state(sa, nbatch)
        gpu_state = CuArray(random_state(sa, nbatch))
        pulse_amplitude = 10.0
        pulse_width = 1.0
        annealing_time = 2000

        track_equilibration_pulse_cpu!(HeatBath(), ExponentialGradient(pulse_amplitude, pulse_width, 1e-5),
        sa, cpu_state, annealing_time;
        accelerate_flip = true)

        track_equilibration_pulse_gpu!(HeatBath(), ExponentialGradient(pulse_amplitude, pulse_width, 1e-5),
        sa, gpu_state, annealing_time;
        accelerate_flip = true)   

        cpu_state_energy = energy(sa, cpu_state)
        gpu_state_energy = energy(sa, gpu_state)
        cpu_success = count(x -> x == 0, cpu_state_energy)
        gpu_success = count(x -> x == 0, gpu_state_energy)
        @info cpu_success, gpu_success
        @test abs((cpu_success - gpu_success) / nbatch) <= 0.03
    end
end