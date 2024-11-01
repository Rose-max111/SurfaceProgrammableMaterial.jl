using Test
using SurfaceProgrammableMaterial: natom, atoms, hasparent
using SurfaceProgrammableMaterial: evaluate_parent, calculate_energy
using SurfaceProgrammableMaterial: parent_nodes, child_nodes
using SurfaceProgrammableMaterial: SimulatedAnnealingHamiltonian
using SurfaceProgrammableMaterial: get_parallel_flip_id

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
        # @info "this_flip_group = $this_flip_group"
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
                # @info "i = $(this_flip_group[i]), j = $(this_flip_group[j])"
                # @info "related_gadgets_i = $related_gadgets_i, related_gadgets_j = $related_gadgets_j"
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
