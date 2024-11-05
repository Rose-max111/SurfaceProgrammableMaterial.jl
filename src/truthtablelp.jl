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

struct IsingGadget{T<:Real}
    logical_spins::Vector{Int}
    J::Vector{T}
    h::Vector{T}
    ground_states::Vector{Int}
end
nspin(ig::IsingGadget) = length(ig.h)
ground_state(ig::IsingGadget, i::Int) = [readbit(ig.ground_states[i], j) for j in 1:nspin(ig)]
function Base.show(io::IO, ig::IsingGadget)
    print(io, typeof(ig))
    print(io, "(
    logical_spins = $(ig.logical_spins),
    J = $(ig.J),
    h = $(ig.h),
    ground_states (little endian) = $(BitStr{nspin(ig)}.(ig.ground_states))
)")
end
Base.show(io::IO, ::MIME"text/plain", ig::IsingGadget) = show(io, ig)

# TODO: this function is not safe to use!
# YM: check this function?
function round_integer!(ig::IsingGadget)
    for v in [ig.J, ig.h]
        v .= round.(v, digits=4)
        v .*= 10^4
    end
    divisior = gcd(vcat(Int.(ig.J), Int.(ig.h)))
    if divisior != 0
        for v in [ig.J, ig.h]
            v .÷= divisior
        end
    end
    return ig
end

function query_model(gate, nspin::Int; round_integer::Bool=true)
    ni = nin(gate)
    @assert nspin >= ni+1 "nspin ($nspin) must be at least $(ni+1)"
    nancilla = nspin - ni - 1
    ground_states0 = [b | (gate(ntuple(i->readbit(b, i), ni)...) << ni) for b in 0:2^ni-1]  # TODO: the order has been changed!
    for ancilla_idx in 0:(2^(2^ni))^nancilla-1  # ancilla configurations: e.g. for ancilla (a, b), the configurations are (b7,..., b1, b0, a7,..., a1, a0)
        ground_states = copy(ground_states0)
        for l = 1:2^ni, k = 1:nancilla
            ancilla_config = readbit(ancilla_idx, 2^ni * (k-1) + l)
            ground_states[l] |= (ancilla_config << (ni + k))
        end
        res = find_ising_gadget(ground_states, setdiff(0:2^nspin-1, ground_states), nspin)
        if res !== nothing
            @assert length(res) == nspin * (nspin - 1) ÷ 2 + nspin
            J, h = res[1:nspin * (nspin - 1) ÷ 2], res[nspin * (nspin - 1) ÷ 2 + 1:end]
            result = IsingGadget(collect(1:ni+1), J, h, ground_states)
            round_integer && round_integer!(result)
            return result
        end
    end
    return nothing
end

function find_ising_gadget(ground_states, excited_states, nspin)
    @assert length(ground_states) >= 1
    @assert length(ground_states) + length(excited_states) == 2^nspin

    # initialize an LP optimizer
    model = Model(HiGHS.Optimizer)
    set_silent(model)  # suppress output
    @variable(model, x[1:nspin * (nspin - 1) ÷ 2 + nspin])  # Ising parameters: Jij and hi
    # ground states are degenerate
    for id in 2:length(ground_states)
        weights = _make_constriant(first(ground_states), ground_states[id], nspin)
        @constraint(model, sum(weights[i] * x[i] for i in eachindex(weights)) == 0)
    end
    # energy of excited states are higher than ground states
    for id in eachindex(excited_states)
        weights = _make_constriant(first(ground_states), excited_states[id], nspin)
        @constraint(model, sum(weights[i] * x[i] for i in eachindex(weights)) <= -1)  # delta = 1
    end

    # start optimization, return nothing if not converged
    optimize!(model)
    if is_solved_and_feasible(model)
        return [value(x[i]) for i in 1:length(x)]
    end
    return nothing
end