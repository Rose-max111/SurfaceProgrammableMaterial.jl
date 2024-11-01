using SurfaceProgrammableMaterial
using GenericTensorNetworks

function write_data(total_atoms, weights, ruleid, msk)
    filename = (pwd()) * "/data/spin_glass_mapping/$(ruleid).txt"
    weights = map(x -> x==-0.0 ? 0.0 : x, weights)
    open(filename, "w") do io
        println(io, "Total Atoms = $total_atoms")
        println(io, "")
        cnt = 0
        for i in 1:total_atoms
            for j in i+1:total_atoms
                cnt = cnt + 1
                println(io,"J($i, $j)=", weights[cnt])
            end
        end
        for i in 1:total_atoms
            println(io, "h($i)=$(weights[cnt+i])")
        end

        println(io, "")
        hyperedges = [[i, j] for i in 1:total_atoms for j in i+1:total_atoms]
        for i in 1:total_atoms
            push!(hyperedges, [i])
        end
        hyperweights = weights
        spgls = SpinGlass(total_atoms, hyperedges, hyperweights)
        spproblem = GenericTensorNetwork(spgls)
        gs = GenericTensorNetworks.solve(spproblem, CountingMin())[]
        println(io, "Ground state energy = $(gs.n), ", "Ground state degenerate = $(gs.c)")

        println(io, "")
        println(io, "The 8 degenerate ground states are:")
        cnt = 0
        for p in [0,1]
            for q in [0,1]
                for r in [0,1]
                    cnt = cnt + 1
                    state = [p, q, r, generic_logic_grate(p, q, r, ruleid)]
                    for i in 1:total_atoms-4
                        push!(state, (msk[i]>>(cnt-1))&1)
                    end
                    state = map(x -> 2*x-1, state)
                    println(io, state)
                end
            end
        end
    end
end


function write_data_readable(total_atoms, weights, ruleid, msk)
    filename = (pwd()) * "/data/spin_glass_mapping/densedata.txt"
    weights = map(x -> x==-0.0 ? 0.0 : x, weights)
    weights = Int.(weights)
    open(filename, "a") do io
        print(io, "$total_atoms ")
        for i in weights
            print(io, "$i ")
        end
        println(io, "")
    end
end

function init()
    folderpath = (pwd()) * "/data/spin_glass_mapping/"
    if !isdir(folderpath)
        mkpath(folderpath)
    end
    for file in readdir(folderpath)
        rm(joinpath(folderpath, file))
    end
end

function __main__()
    init()
    natoms = Vector{Vector{Int}}()
    previous = Vector{Int}()
    for total_atoms = 4:1:5
        ok = Vector{Int}()
        for id in 0:255
            if (id in previous) == false
                msk, weights = query_model(id, total_atoms)
                if msk != -1
                    push!(ok, id)
                    push!(previous, id)
                    @info "now testing ruleid = $id, total_atoms = $total_atoms, weights = $(weights)"
                    # check_vaild(total_atoms, weights, id, msk)
                    write_data(total_atoms, weights, id, msk)
                    write_data_readable(total_atoms, weights, id, msk)
                end
            end
        end
        push!(natoms, ok)
    end
end

__main__()
