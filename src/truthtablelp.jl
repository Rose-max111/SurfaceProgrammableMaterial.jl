# let state1 - state2 = val
function _make_constriant(state1::Integer, state2::Integer, total_atoms::Int)
    takespin(b::Integer, i::Int) = 2 * readbit(b, i) - 1
    weights = Int[]
    for i in 1:total_atoms
        for j in i+1:total_atoms
            push!(weights, takespin(state1, i) * takespin(state1, j) - takespin(state2, i) * takespin(state2, j))
        end
    end
    for i in 1:total_atoms
        push!(weights, takespin(state1, i) - takespin(state2, i))
    end
    return weights
end

function set_value(configuration::Integer, ancilla_mask::Integer, ancilla_configuration::Integer)
    return (configuration & (~ancilla_mask)) | ancilla_configuration
end

# subspins: the spins to implement the rule, the rest are ancilla
function find_proper_model(::CellularAutomata1D{N}, total_atoms::Int, logical_spins) where N
    @assert length(logical_spins) == 4 && total_atoms >= 4 "length(logical_spins) ($logical_spins) must be 4 and total_atoms ($total_atoms) must be at least 4"
    mask_ancilla = bmask(Int, length(logical_spins)+1:total_atoms)
    ground_states = collect(0:2^length(logical_spins)-1)
    for ancilla_idx in 0:2^(total_atoms-length(logical_spins))-1
        set_value.(ground_states, mask_ancilla, ancilla_idx << length(logical_spins))
        res = find_ising_gadget(ground_states, setdiff(1:total_atoms, logical_spins), total_atoms, logical_spins, ((a, b, c, d)->automatarule(a, b, c, ruleid) == d))
        if res !== nothing
            return mask, res
        end
    end
    return -1, false
end

function find_ising_gadget(ground_states, excited_states, total_atoms, logical_spins, constraint)
    @assert length(ground_states) + length(excited_states) == 2^total_atoms
    #model = @suppress Model(HiGHS.Optimizer)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:Int(total_atoms * (total_atoms - 1) / 2 + total_atoms)])  # Jij and hi
    delta = 1
    for id in 2:length(ground_states)
        weights = _make_constriant(ground_states[1], ground_states[id], total_atoms)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) == 0)
    end

    for id in 1:length(excited_states)
        weights = _make_constriant(ground_states[1], excited_states[id], total_atoms)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) <= -delta)
    end

    # @objective(model, Min, x[1])
    optimize!(model)
    if is_solved_and_feasible(model)
        return [value(x[i]) for i in 1:length(x)]
    else
        @info "constraint = $constraint, failed to find a solution"
    end
    for con in all_constraints(model; include_variable_in_set_constraints = false)
        delete(model, con)
    end
    return nothing
end


function query_model(rule::CellularAutomata1D, total_atoms)
    mask, weights = find_proper_model(rule, total_atoms, [1,2,3,4])
    if mask == -1
        return -1, false
    end
    weights = round.(weights, digits=2)
    weights = weights ./ 0.25
    return mask, weights
end