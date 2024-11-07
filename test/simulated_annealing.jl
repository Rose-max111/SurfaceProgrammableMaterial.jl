using Test, CUDA
using SurfaceProgrammableMaterial
using SurfaceProgrammableMaterial: nspin, spins
using SurfaceProgrammableMaterial: unsafe_energy_temperature
using SurfaceProgrammableMaterial: unsafe_parent_nodes, unsafe_child_nodes
using SurfaceProgrammableMaterial: SimulatedAnnealingHamiltonian
using SurfaceProgrammableMaterial: parallel_scheme
using SurfaceProgrammableMaterial: random_state
using SurfaceProgrammableMaterial: update_temperature!
using SurfaceProgrammableMaterial: SARuntime_CUDA

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

if CUDA.functional()
    @testset "temperature gradient gpu" begin
        sa = SimulatedAnnealingHamiltonian(8, 12, CellularAutomata1D(110))
        nbatch = 200
        runtime = 200
        
        r_cpu = SARuntime(Float64, sa, nbatch)
        eg = ExponentialGradient(1.0, 1.0, 1e-5)
        update_temperature!(r_cpu, eg, 40, runtime, false)

        r_gpu = CUDA.cu(r_cpu)
        update_temperature!(r_gpu, eg, 40, runtime, false)

        @test r_cpu.temperature ≈ Array(r_gpu.temperature)
        # @info r_cpu.temperature
        # @info Array(r_gpu.temperature)
    end
end

@testset "energy_calculation" begin
    # 0001
    # 1110
    # 0110
    # 1011
    sa = SimulatedAnnealingHamiltonian(4, 4, CellularAutomata1D(110))
    state = reshape(Bool[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1], 16, 1)
    ibatch = 1
    @test length(state) == nspin(sa)
    @test unsafe_energy_temperature(sa, state, ones(Float64, size(state)), 5, ibatch)[1] == 1
    @test unsafe_energy_temperature(sa, state, ones(Float64, size(state)), 6, ibatch)[1] == 0
    @test unsafe_energy_temperature(sa, state, ones(Float64, size(state)), 7, ibatch)[1] == 0
    @test unsafe_energy_temperature(sa, state, ones(Float64, size(state)), 8, ibatch)[1] == 0
    @test unsafe_energy_temperature(sa, state, ones(Float64, size(state)), 10, ibatch)[1] == 0
    @test energy(sa, state)[ibatch] == 3
end

@testset "parent_nodes_child_nodes" begin
    sa = SimulatedAnnealingHamiltonian(5, 5, CellularAutomata1D(110))
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
    each_flip_group = parallel_scheme(sa)
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
    end
end

@testset "pulse_equilibration_cpu" begin
    sa = SimulatedAnnealingHamiltonian(8, 8, CellularAutomata1D(110))
    nbatch = 2000
    annealing_time = 200

    eg = ExponentialGradient(2.0, 2.0, 1e-5)
    tracker = SAStateTracker()
    r = SARuntime(Float64, sa, nbatch)
    track_equilibration_pulse!(r, eg, annealing_time; flip_scheme = 1:nspin(sa), tracker)
    @test length(tracker.state) == length(tracker.temperature) == 201
 
    state_energy = energy(sa, r.state)
    success = count(x -> x == 0, state_energy)
    @show success / nbatch
    @test abs((success / nbatch) - 0.45) <= 0.1

    r = SARuntime(Float64, sa, nbatch)
    track_equilibration_pulse!(r, eg, annealing_time; flip_scheme = parallel_scheme(sa))
    sequential_flip_state_energy = energy(sa, r.state)
    sequential_flip_success = count(x -> x==0, sequential_flip_state_energy)
    @show success, sequential_flip_success
    @test abs((sequential_flip_success - success) / nbatch) <= 0.05
end

if CUDA.functional()
    @testset "pulse_equilibration_gpu" begin
        sa = SimulatedAnnealingHamiltonian(11, 7, CellularAutomata1D(110))
        nbatch = 2000
        annealing_time = 600
    
        eg = ExponentialGradient(5.0, 1.7, 1e-5)
        r_cpu = SARuntime(Float64, sa, nbatch)
        track_equilibration_pulse!(r_cpu, eg, annealing_time; flip_scheme = 1:nspin(sa))
        state_energy_cpu = energy(sa, r_cpu.state)
        success_cpu = count(x -> x == 0, state_energy_cpu)
        @show success_cpu / nbatch

        r_gpu = CUDA.cu(r_cpu)
        track_equilibration_pulse!(r_gpu, eg, annealing_time; flip_scheme = 1:nspin(sa))
        state_energy_gpu = energy(sa, Array(r_gpu.state))
        success_gpu = count(x -> x == 0, state_energy_gpu)
        @show success_gpu / nbatch

        @test abs((success_cpu / nbatch) - (success_gpu / nbatch)) <= 0.05

        r_gpu_parallel = CUDA.cu(r_cpu)
        track_equilibration_pulse!(r_gpu_parallel, eg, annealing_time; flip_scheme = CUDA.cu.(parallel_scheme(sa)))
        state_energy_gpu_parallel = energy(sa, Array(r_gpu_parallel.state))
        success_gpu_parallel = count(x -> x == 0, state_energy_gpu_parallel)
        @show success_gpu_parallel / nbatch
        @show success_cpu, success_gpu, success_gpu_parallel
        @test abs((success_gpu_parallel / nbatch) - (success_gpu / nbatch)) <= 0.05
    end
end
