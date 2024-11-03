using Test
using SurfaceProgrammableMaterial
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
    cnt = 0
    for p in [0, 1]
        for q in [0, 1]
            for r in [0, 1]
                cnt += 1
                state = [p, q, r, automatarule(p, q, r, ruleid)]
                for i in 1:total_atoms-4
                    push!(state, (msk[i]>>(cnt-1))&1)
                end
                state .⊻= 1
                stbit = StaticBitVector(state)
                this_energy = spinglass_energy(spgls, stbit)
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
    end
end


@testset "set_value" begin
    @test SurfaceProgrammableMaterial.set_value(0b1010, 0b1100, 0b0100) == 0b0110
    @test SurfaceProgrammableMaterial.set_value(0b1010, 0b1001, 0b1001) == 0b1011
end

@testset "query_model" begin
    natoms = Vector{Vector{Int}}()
    previous = Vector{Int}()
    for total_atoms = 4:1:5
        ok = Vector{Int}()
        for id in 0:255
            if (id in previous) == false
                msk, weights = query_model(CellularAutomata1D(id), total_atoms)
                if msk != -1
                    push!(ok, id)
                    push!(previous, id)
                    @info "now testing ruleid = $id, total_atoms = $total_atoms, weights = $(weights)"
                    check_vaild(total_atoms, weights, id, msk)
                end
            end
        end
        push!(natoms, ok)
    end
end