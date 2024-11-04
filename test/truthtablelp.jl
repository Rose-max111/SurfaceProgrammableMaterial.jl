using Test
using SurfaceProgrammableMaterial, BitBasis
using GenericTensorNetworks

function check_vaild(total_atoms, weights, ruleid, msk)
    hyperedges = [[i, j] for i in 1:total_atoms for j in i+1:total_atoms]
    for i in 1:total_atoms
        push!(hyperedges, [i])
    end
    hyperweights = weights
    spgls = SpinGlass(total_atoms, hyperedges, hyperweights)
    spproblem = GenericTensorNetwork(spgls)
    gs = GenericTensorNetworks.solve(spproblem, CountingMin())[]
    @test gs.c == 8.0
    for k = 0:7
        p, q, r = readbit(k, 1), readbit(k, 2), readbit(k, 3)
        state = [p, q, r, automatarule(p, q, r, ruleid)]
        for i in 1:total_atoms-4
            push!(state, (msk[i]>>k)&1)
        end
        this_energy = spinglass_energy(spgls, state)
        # @info "state = $(state.⊻1), energy = $this_energy"
        @test this_energy == gs.n
        my_energy = 0.0
        iiid = 0
        for i in 1:total_atoms
            for j in i+1:total_atoms
                iiid += 1
                my_energy += weights[iiid] * (state[i] == 1 ? -1 : 1) * (state[j] == 1 ? -1 : 1)
            end
        end
        for i in 1:total_atoms
            my_energy += weights[iiid+i] * (state[i] == 1 ? -1 : 1)
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
                    check_vaild(total_atoms, vcat(res.J, res.h), id, res.ancilla_idx)
                end
            end
        end
        push!(natoms, ok)
    end
end