using Test
using SurfaceProgrammableMaterial, BitBasis
using GenericTensorNetworks

function check_vaild(ig::IsingGadget, ruleid)
    # check ground states
    a, b, c, d = ig.logical_spins
    for i=1:length(ig.ground_states)
        state = ground_state(ig, i)
        @test automatarule(state[a], state[b], state[c], ruleid) == state[d]
    end

    # check model parameters
    hyperedges = [[i, j] for i in 1:nspin(ig) for j in i+1:nspin(ig)]
    for i in 1:nspin(ig)
        push!(hyperedges, [i])
    end
    hyperweights = vcat(ig.J, ig.h)
    spgls = SpinGlass(nspin(ig), hyperedges, hyperweights)
    spproblem = GenericTensorNetwork(spgls)
    gs = GenericTensorNetworks.solve(spproblem, CountingMin())[]
    @test gs.c == 8.0
    for k = 0:7
        state = ground_state(ig, k+1)
        this_energy = spinglass_energy(spgls, state)
        # @info "state = $(state.⊻1), energy = $this_energy"
        @test this_energy == gs.n
        my_energy = 0.0
        iiid = 0
        for i in 1:nspin(ig)
            for j in i+1:nspin(ig)
                iiid += 1
                my_energy += ig.J[iiid] * (state[i] == 1 ? -1 : 1) * (state[j] == 1 ? -1 : 1)
            end
        end
        for i in 1:nspin(ig)
            my_energy += ig.h[i] * (state[i] == 1 ? -1 : 1)
        end
        @test my_energy == gs.n
    end
end

@testset "query_model" begin
    natoms = Vector{Vector{Int}}()
    previous = Vector{Int}()
    for total_atoms = 4:1:5
        ok = Vector{Int}()
        for id in 0:255
            if (id in previous) == false
                res = query_model(CellularAutomata1D(id), total_atoms)
                if res !== nothing
                    push!(ok, id)
                    push!(previous, id)
                    @info "now testing ruleid = $id, total_atoms = $total_atoms"
                    check_vaild(res, id)
                end
            end
        end
        push!(natoms, ok)
    end
end