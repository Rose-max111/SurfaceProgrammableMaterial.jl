# let state1 - state2 = val
function _make_constriant(state1::Integer, state2::Integer, nspin::Int)
    takespin(b::Integer, i::Int) = 1 - 2 * readbit(b, i)
    weights = Int[]
    for i in 1:nspin
        for j in i+1:nspin
            push!(weights, takespin(state1, i) * takespin(state1, j) - takespin(state2, i) * takespin(state2, j))
        end
    end
    for i in 1:nspin
        push!(weights, takespin(state1, i) - takespin(state2, i))
    end
    return weights
end

# subspins: the spins to implement the rule, the rest are ancilla
function find_proper_model(rule::CellularAutomata1D{N}, nspin::Int) where N
    @assert nspin >= 4 "nspin ($nspin) must be at least 4"
    nancilla = nspin - 4
    ground_states0 = [b | (rule(readbit(b, 1), readbit(b, 2), readbit(b, 3)) << 3) for b in 0:2^3-1]  # TODO: the order has been changed!
    @assert length(ground_states0) == 8
    for ancilla_idx in 0:256^nancilla-1  # ancilla configurations: e.g. for ancilla (a, b), the configurations are (b7,..., b1, b0, a7,..., a1, a0)
        ground_states = copy(ground_states0)
        for l = 1:8, k = 1:nancilla
            ancilla_config = readbit(ancilla_idx, 8 * (k-1) + l)
            ground_states[l] |= (ancilla_config << (4 + k - 1))
        end
        res = find_ising_gadget(ground_states, setdiff(0:2^nspin-1, ground_states), nspin)
        if res !== nothing
            return ancilla_idx, res
        end
    end
    return -1, false
end

function find_ising_gadget(ground_states, excited_states, nspin)
    @assert length(ground_states) + length(excited_states) == 2^nspin
    #model = @suppress Model(HiGHS.Optimizer)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:Int(nspin * (nspin - 1) / 2 + nspin)])  # Jij and hi
    for id in 2:length(ground_states)
        weights = _make_constriant(ground_states[1], ground_states[id], nspin)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) == 0)
    end

    for id in 1:length(excited_states)
        weights = _make_constriant(ground_states[1], excited_states[id], nspin)
        @constraint(model, sum(weights[i] * x[i] for i in 1:length(weights)) <= -1)  # delta = 1
    end

    # @objective(model, Min, x[1])
    optimize!(model)
    if is_solved_and_feasible(model)
        return [value(x[i]) for i in 1:length(x)]
    end
    return nothing
end


function query_model(rule::CellularAutomata1D, nspin)
    ancilla_idx, weights = find_proper_model(rule, nspin)
    if ancilla_idx == -1
        return -1, false
    end
    weights = round.(weights, digits=2)
    weights = weights ./ 0.25
    return [ancilla_idx], weights
end
