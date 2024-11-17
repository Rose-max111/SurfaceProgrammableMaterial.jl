struct SimulatedAnnealingHamiltonian{ET}
    n::Int # number of spins per layer
    m::Int # number of layers in the time direction
    energy_term::ET  # e.g. a cellular automaton rule, takes 3 inputs and returns 1 output
    periodic::Bool  # whether to use periodic boundary condition
end
SimulatedAnnealingHamiltonian(n::Int, m::Int, energy_term) = SimulatedAnnealingHamiltonian(n, m, energy_term, true)

nspin(sa::SimulatedAnnealingHamiltonian) = sa.n * sa.m
spins(sa::SimulatedAnnealingHamiltonian) = Base.OneTo(nspin(sa))
function random_state(sa::SimulatedAnnealingHamiltonian, nbatch::Integer)
    return rand(Bool, nspin(sa), nbatch)
end
# evaluate the energy of the state, the batched version
function energy(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix)
    return [sum(i->unsafe_energy(sa, state, i, ibatch), sa.n+1:nspin(sa)) for ibatch in 1:size(state, 2)]
end

# NOTE: MT and ST can be CuArray type.
struct SARuntime{T, ET, MT<:AbstractMatrix{T}, ST<:AbstractMatrix{Bool}}    # runtime information for simulated annealing
    hamiltonian::SimulatedAnnealingHamiltonian{ET}   # the Hamiltonian
    state::ST    # state
    temperature::MT    # temperature, which has the same size as state
    function SARuntime(hamiltonian::SimulatedAnnealingHamiltonian{ET}, state::AbstractMatrix{Bool}, temperature::AbstractMatrix{T}) where {T, ET}
        @assert size(temperature) == size(state) "temperature and state must have the same size"
        @assert nspin(hamiltonian) == size(state, 1) "state must have the same number of spins as the Hamiltonian"
        new{T, ET, typeof(temperature), typeof(state)}(hamiltonian, state, temperature)
    end
end
nbatch(sr::SARuntime) = size(sr.state, 2)
# initialize the runtime information with random state
function SARuntime(::Type{T}, sa::SimulatedAnnealingHamiltonian{ET}, nbatch::Integer) where {T, ET}
    return SARuntime(sa, random_state(sa, nbatch), ones(T, nspin(sa), nbatch))
end

@inline function unsafe_energy(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, inode::Integer, ibatch::Integer)
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        i, j = CartesianIndices((sa.n, sa.m))[inode].I
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        return trueoutput ⊻ state[inode, ibatch]
    end
end

@inline function unsafe_step_kernel!(transition_rule::TransitionRule,
        sa::SimulatedAnnealingHamiltonian{ET},
        state::AbstractMatrix{Bool},
        temperature::AbstractMatrix{T},
        ibatch::Integer,
        node::Integer
    ) where {T, ET<:CellularAutomata1D}
    @inbounds begin
        grid = CartesianIndices((sa.n, sa.m))
        i, j = grid[node].I  # proposed position

        ΔE_over_T_previous_layer = j > 1 ? ((ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch); (one(T) - 2 * ΔE) / Teff) : zero(T)
        ΔE_over_T_next_layer = zero(T)
        if j < sa.m # not the last layer
            # calculate the energy from the child nodes
            cnodes = unsafe_child_nodes(sa, node)
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer -= ΔE / Teff
            end
            # calculate the energy from the child nodes after flipping
            state[node, ibatch] ⊻= true
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer += ΔE / Teff
            end
            state[node, ibatch] ⊻= true
        end
        ΔE_over_T = ΔE_over_T_previous_layer + ΔE_over_T_next_layer
        if rand() < prob_accept(transition_rule, ΔE_over_T)
            state[node, ibatch] ⊻= true
        end
    end
end
@inline function unsafe_energy_temperature(sa::SimulatedAnnealingHamiltonian{ET}, state::AbstractMatrix{Bool}, temperature::AbstractMatrix{T}, inode::Integer, ibatch::Integer) where {T, ET<:CellularAutomata1D}
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        i, j = CartesianIndices((sa.n, sa.m))[inode].I
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        # NOTE: two sqrt is much faster than x^0.25
        Teff = sqrt(sqrt(temperature[a, ibatch] * temperature[b, ibatch] * temperature[c, ibatch] * temperature[inode, ibatch]))
        return trueoutput ⊻ state[inode, ibatch], Teff
    end
end
# NOTE: this function does not perform boundary check
@inline function unsafe_parent_nodes(sa::SimulatedAnnealingHamiltonian{<:CellularAutomata1D}, node::Integer)
    grid = CartesianIndices((sa.n, sa.m))
    lis = LinearIndices((sa.n, sa.m))
    @inbounds i, j = grid[node].I
    @inbounds sa.periodic ? (lis[mod1(i-1, sa.n), j-1], lis[i, j-1], lis[mod1(i+1, sa.n), j-1]) : (lis[i-1, j-1], lis[i, j-1], lis[i+1, j-1])
end
@inline function unsafe_child_nodes(sa::SimulatedAnnealingHamiltonian{<:CellularAutomata1D}, node::Integer)
    grid = CartesianIndices((sa.n, sa.m))
    lis = LinearIndices((sa.n, sa.m))
    @inbounds i, j = grid[node].I
    @inbounds sa.periodic ? (lis[mod1(i-1, sa.n), j+1], lis[mod1(i, sa.n), j+1], lis[mod1(i+1, sa.n), j+1]) : (lis[i-1, j+1], lis[i, j+1], lis[i+1, j+1])
end

function parallel_scheme(sa::SimulatedAnnealingHamiltonian{<:CellularAutomata1D}; input_fixed=false)
    ret = Vector{Vector{Int}}()
    for cnt in 1:6
        temp = Vector{Int}()
        for layer in 1+div(cnt - 1, 3):2:sa.m
            for position in mod1(cnt, 3):3:(sa.n - sa.n % 3)
                push!(temp, LinearIndices((sa.n, sa.m))[position, layer])
            end
        end
        push!(ret, temp)
    end
    if sa.n % 3 >= 1
        push!(ret, collect(sa.n:2*sa.n:sa.n*sa.m))
        push!(ret, collect(2*sa.n:2*sa.n:sa.n*sa.m))
    end
    if sa.n % 3 >= 2
        push!(ret, collect(sa.n-1:2*sa.n:sa.n*sa.m))
        push!(ret, collect(2*sa.n-1:2*sa.n:sa.n*sa.m))
    end
    input_fixed == false && return ret
    for i in 1:length(ret)
        ret[i] = setdiff(ret[i], collect(1:sa.n))
    end
    return ret
end

struct SAStateTracker{T, MT<:AbstractMatrix{T}, ST<:AbstractMatrix{Bool}}
    state::Vector{ST}
    temperature::Vector{MT}
end
SAStateTracker() = SAStateTracker(Vector{Matrix{Bool}}(), Vector{Matrix{Float64}}())
function track!(tracker::SAStateTracker, state, temperature)
    push!(tracker.state, state)
    push!(tracker.temperature, temperature)
end

# update the temperature
# - temprule: the temperature gradient
# - t: the current step index
# - annealing_time: the total number of steps
function update_temperature!(runtime::SARuntime, temprule::ColumnWiseGradient, t::Integer, annealing_time::Integer, reverse_direction::Bool)
    dcut = cutoff_distance(temprule)
    sa = runtime.hamiltonian
    each_movement = (dcut * 2 + sa.m) / (annealing_time - 1)
    middle_position = reverse_direction ? sa.m + dcut - t * each_movement : -dcut + t * each_movement
    temperature_matrix!(reshape(view(runtime.temperature, :, 1), sa.n, sa.m), temprule, 1:sa.m, middle_position)
    view(runtime.temperature, :, 2:size(runtime.temperature, 2)) .= view(runtime.temperature, :, 1:1)
end
function update_temperature!(runtime::SARuntime, temprule::TemperatureCollective, t::Integer, annealing_time::Integer, reverse_direction::Bool)
    proportion = t / annealing_time
    this_temperature = evaluate_temperature(temprule, proportion)
    view(runtime.temperature, :, 1) .= this_temperature
    view(runtime.temperature, :, 2:size(runtime.temperature, 2)) .= view(runtime.temperature, :, 1:1)
end
function temperature_matrix!(output::AbstractMatrix, tg::ColumnWiseGradient, offsets::AbstractArray, middle_position::Real)
    offsets = _match_device(output, offsets)
    t = evaluate_temperature.(Ref(tg), offsets .- middle_position)
    output .= reshape(t, 1, :)  # broadcast assignment
end
_match_device(::AbstractMatrix, offsets) = offsets
_flip_match_device(::SARuntime, spins) = spins

function track_equilibration_pulse!(
                runtime::SARuntime,
                temprule::TemperatureRule,
                annealing_time;
                tracker = nothing,
                flip_scheme = 1:nspin(runtime.hamiltonian),
                transition_rule::TransitionRule = HeatBath(),
                reverse_direction::Bool=false
            )
    sa = runtime.hamiltonian
    # @assert all(sort!(vcat(flip_scheme...)) .== 1:nspin(sa)) "invalid flip scheme: $flip_scheme, must be a perfect cover of all spins: $(1:nspin(sa))"

    update_temperature!(runtime, temprule, 0, annealing_time, reverse_direction)
    tracker !== nothing && track!(tracker, copy(runtime.state), copy(runtime.temperature))
    for t in 1:annealing_time
        update_temperature!(runtime, temprule, t, annealing_time, reverse_direction)
        for spins in flip_scheme
            # step on a subset of nodes and all batches
            step!(runtime, transition_rule, _flip_match_device(runtime, spins))
        end
        tracker !== nothing && track!(tracker, copy(runtime.state), copy(runtime.temperature))
    end
end

function step!(runtime::SARuntime, transition_rule::TransitionRule, simutanuous_flip_spins)
    for ibatch in 1:size(runtime.state, 2), spin in simutanuous_flip_spins
        unsafe_step_kernel!(transition_rule, runtime.hamiltonian, runtime.state, runtime.temperature, ibatch, spin)
    end
end

# examine the state and explain why the SA fails (if any).
function why(r::SARuntime)
    sim, state, temperature = r.hamiltonian, r.state, r.temperature
    grid = CartesianIndices((sim.n, sim.m))
    for ibatch in 1:size(r.state, 2)
        @info "analyzing batch $ibatch"
        for ispin in sim.n+1:nspin(sim)
            local_energy = unsafe_energy(sim, state, ispin, ibatch)
            i, j = grid[ispin].I
            if local_energy != 0  # an error dectected!
                @info "error detected at spin $((i, j)), batch $ibatch"
                a, b, c = unsafe_parent_nodes(sim, ispin)
                sa, sb, sc = Int(state[a, ibatch]), Int(state[b, ibatch]), Int(state[c, ibatch])
                ca, cb, cc = grid[a].I, grid[b].I, grid[c].I
                @info "- parents: $ca = $sa, $cb = $sb, $cc = $sc"
                expected_state = sim.energy_term(sa, sb, sc)
                @info "- got: $(Int(state[ispin, ibatch])) instead of $expected_state"
                children = unsafe_child_nodes(sim, ispin)
                nc = 0
                for child in children
                    parents = unsafe_parent_nodes(sim, child)
                    if sim.energy_term(map(p->p==ispin ? !state[ispin, ibatch] : state[p, ibatch], parents)...) != state[child, ibatch]
                        nc += 1
                    end
                end
                @info "- by flipping it, $nc children will be violated"
            end
        end
    end
end