
J = [0 1 1 2 3; 1 0 2 2 5; 1 2 0 2 5; 2 2 2 0 6; 3 5 5 6 0]
h = [1, 2, 2, 2, 5]

ti = 3
for i in 1:3
    J[i, 4] *= ti^2
    J[4, i] *= ti^2
    J[5, i] *= ti
    J[i, 5] *= ti
end
h[4] *= ti^2
h[5] *= ti

function energy(J, h, state)
    n = length(state)
    energy = 0
    for i in 1:n
        energy += h[i] * state[i]
        for j in i+1:n
            energy += J[i, j] * state[i] * state[j]
        end
    end
    return energy
end

true_config = [7, 10, 11, 12, 13, 14, 16, 17]
for bit_config in true_config
    state = [((bit_config >> i) & 1) * 2 - 1 for i in 0:4]
    # if energy(J, h, state) == -11
    #     @show bit_config
    #     for i in 1:5
    #         for j in max(i+1, 4):5
    #             if J[i, j] * state[i] * state[j] > 0
    #                 println("Frustated bond: ", i, " ", j)
    #             end
    #         end 
    #     end
    # end
    e1 = energy(J, h, state)
    state[5] = -state[5]
    e2 = energy(J, h, state)
    state[4] = -state[4]
    e3 = energy(J, h, state)
    @show bit_config, e1, e2, e3
end

total_energy = []
for bit_config in 0:2^5-1
    state = [((bit_config >> i) & 1) * 2 - 1 for i in 0:4]
    push!(total_energy, energy(J, h, state))
end
sort!(total_energy)
@show total_energy[1:8]