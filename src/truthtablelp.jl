# let state1 - state2 = val
function make_constriant(state1, state2, total_atoms)
    state1_spin = map(x->2*x-1, state1)
    state2_spin = map(x->2*x-1, state2)
    weights = []
    for i in 1:total_atoms
        for j in i+1:total_atoms
            push!(weights, state1_spin[i]*state1_spin[j] - state2_spin[i]*state2_spin[j])
        end
    end
    for i in 1:total_atoms
        push!(weights, state1_spin[i] - state2_spin[i])
    end
    return weights
end

function generate_vaild_mask!(mask_list, now, endpos, now_mask)
    if now == endpos+1
        push!(mask_list, copy(now_mask))
        return mask_list
    end
    for i in 0:255
        now_mask[now] = i
        generate_vaild_mask!(mask_list, now+1, endpos, now_mask)
    end
    return mask_list
end

function find_proper_model(subset, ruleid, total_atoms)
    total_mask = Vector{Vector{Int}}()
    generate_vaild_mask!(total_mask, 1, total_atoms - 4, zeros(Int, total_atoms - 4))
    for mask in total_mask 
        res = find_ising_gadget(mask, subset, ((a, b, c, d)->automatarule(a, b, c, ruleid) == d), total_atoms)
        if res !== nothing
            return mask, res
        end
    end
    return -1, false
end

function find_ising_gadget(mask, subset, constraint, total_atoms)
    #model = @suppress Model(HiGHS.Optimizer)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:Int(total_atoms * (total_atoms - 1) / 2 + total_atoms)])  # Jij and hi
    delta = 1
    cnt = 0
    proper_states = Vector{Vector{Int}}()

    for p in [0,1]
        for q in [0,1]
            for r in [0,1]
                cnt += 1
                st = [p, q, r, constraint(p, q, r)]
                for t in mask
                    push!(st, (t>>(cnt-1)) & 1)
                end
                push!(proper_states, st)
            end
        end
    end

    wrong_states = Vector{Vector{Int}}()
    for thismsk in 0:2^total_atoms-1
        state = [thismsk>>i&1 for i in 0:total_atoms-1]
        if in(state, proper_states) == false
            push!(wrong_states, state)
        end
    end
    @assert length(wrong_states) + length(proper_states) == 2^total_atoms
    
    for id in 2:length(proper_states)
        weights = make_constriant(proper_states[1], proper_states[id], total_atoms)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) == 0)
    end

    for id in 1:length(wrong_states)
        weights = make_constriant(proper_states[1], wrong_states[id], total_atoms)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) <= -delta)
    end

    # @objective(model, Min, x[1])
    optimize!(model)
    if is_solved_and_feasible(model)
        return mask, [value(x[i]) for i in 1:length(x)]
    else
        @info "mask = $mask, constraint = $constraint, failed to find a solution"
    end
    for con in all_constraints(model; include_variable_in_set_constraints = false)
        delete(model, con)
    end
    return nothing
end


function query_model(ruleid, total_atoms)
    mask, weights = find_proper_model([1,2,3,4], ruleid, total_atoms)
    if mask == -1
        return -1, false
    end
    weights = round.(weights, digits=2)
    weights = weights ./ 0.25
    return mask, weights
end