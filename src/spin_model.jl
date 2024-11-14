struct IsingModel{ET}
    nspin::Int
    last_interaction_index::AbstractArray{Int}
    interactions::AbstractMatrix{ET}
    onsites::AbstractArray{ET}
end
nspin(model::IsingModel) = model.nspin

function IsingModel(coupling::Matrix{T}) where T # Here assume coupling[i, i] = 0 ∀ i
    nspin = size(coupling, 1)
    model = IsingModel{T}(nspin, zeros(Int, nspin), zeros(T, nspin * (nspin - 1), 4), zeros(T, nspin))
    nedges_already_constructed = 0
    for i in 1:nspin, j in i+1:nspin
        nedges_already_constructed = add_interaction!(model, i, j, coupling[i, j], nedges_already_constructed)
    end
    @assert nedges_already_constructed == nspin * (nspin - 1) "must construct all edges"
    return model
end

function energy(model::IsingModel, state::AbstractArray)
    ret = 0
    for i in 1:2:size(model.interactions, 1)
        sx, sy = state[Int(model.interactions[i, 1])], state[Int(model.interactions[i, 2])]
        ret += sx ⊻ sy ? -model.interactions[i, 3] : model.interactions[i, 3]
    end
    ret += sum(model.onsites[i] * (state[i] ? 1 : -1) for i in 1:model.nspin)
    return ret
end
function energy(model::IsingModel, state::AbstractMatrix)
    return [energy(model, state[:, i]) for i in 1:size(state, 2)]
end

function random_state(model::IsingModel, nbatch::Integer)
    return rand(Bool, model.nspin, nbatch)
end

function bool_to_spin(state::AbstractArray) # bool representation to spin representation
    return [state[i] == true ? 1 : -1 for i in 1:length(state)]
end
function spin_to_bool(state::AbstractArray) # spin representation to bool representation
    return [state[i] == -1 ? false : true for i in 1:length(state)]
end

function add_interaction!(model::IsingModel, sx::Int, sy::Int, energy::ET, nedges::Int) where ET
    nedges += 1
    model.interactions[nedges, :] = [sx, sy, energy, model.last_interaction_index[sx]]
    model.last_interaction_index[sx] = nedges
    nedges += 1
    model.interactions[nedges, :] = [sy, sx, energy, model.last_interaction_index[sy]]
    model.last_interaction_index[sy] = nedges
    return nedges
end

function spin_model_construction(::Type{T}, sa::SimulatedAnnealingHamiltonian) where T # this time rule 110 only
    nspin_exancilla = sa.n * sa.m
    nspin = nspin_exancilla + sa.n * (sa.m-1) # here we assume each gadget has an ancilla
    lis = LinearIndices((sa.n, sa.m))

    # coefficients of the ising model
    interacting_energy_matrix = Float64.([0 1 1 2 3;1 0 2 2 5;1 2 0 2 5;2 2 2 0 6;3 5 5 6 0])
    onsite_energy = Float64.([1, 2, 2, 2, 5])

    # calculate the number of edges
    howmany_edges = 0
    for i in 1:sa.n, j in 1:sa.m-1
        if sa.periodic == false && (i == 1 || i == sa.n)
            continue
        end
        howmany_edges += 10 * 2 # each interaction is decomposed into 2 directed edges
    end
    
    # construct the mapped Ising model
    already_constructed_edges = 0
    model = IsingModel{T}(nspin, zeros(Int, nspin), zeros(T, howmany_edges, 4), zeros(T, nspin))
    for i in 1:sa.n, j in 1:sa.m-1
        # spin_id listed as [left, center, right, next, acilla]
        spin_id = [lis[mod1(i-1, sa.n), j], lis[i, j], lis[mod1(i+1, sa.n), j], lis[i, j+1], lis[i, j] + nspin_exancilla]
        if sa.periodic == false && (i == 1 || i == sa.n)
            continue
        end
        # @show spin_id, i, j
        for u in 1:5, v in u+1:5
            already_constructed_edges = add_interaction!(model, spin_id[u], spin_id[v], interacting_energy_matrix[u, v], already_constructed_edges)
        end
        for u in 1:5
            model.onsites[spin_id[u]] += onsite_energy[u]
        end
    end
    return model
end

struct SpinSARuntime{T, ET, MT<:AbstractMatrix{T}, ST<:AbstractMatrix{Bool}}
    hamiltonian::SimulatedAnnealingHamiltonian{ET}
    model::IsingModel{T}
    state::ST # Note here 0->↓(sx = -1) 1->↑(sx = 1)
    temperature::MT
    function SpinSARuntime(hamiltonian::SimulatedAnnealingHamiltonian{ET}, model::IsingModel{T}, state::AbstractMatrix{Bool}, temperature::AbstractMatrix{T}) where {T, ET}
        new{T, ET, typeof(temperature), typeof(state)}(hamiltonian, model, state, temperature)
    end
end
nbatch(sr::SpinSARuntime) = size(sr.state, 2)
function SpinSARuntime(::Type{T}, sa::SimulatedAnnealingHamiltonian{ET}, nbatch::Integer) where {T, ET}
    model = spin_model_construction(T, sa)
    return SpinSARuntime(sa, model, random_state(model, nbatch), ones(T, model.nspin, nbatch))
end
function SpinSARuntime(::Type{T}, nbatch::Integer, model::IsingModel{T}) where T
    return SpinSARuntime(SimulatedAnnealingHamiltonian(0, 0, CellularAutomata1D(0)), model, random_state(model, nbatch), ones(T, model.nspin, nbatch))
end

@inline function unsafe_energy(state::AbstractMatrix,
                            interactions::AbstractMatrix, onsites::AbstractArray, last_interaction_index::AbstractArray,
                             inode::Integer, ibatch::Integer)
    last_interaction = last_interaction_index[inode]
    energy = state[inode, ibatch] == true ? onsites[inode] : -onsites[inode]
    while last_interaction != 0
        interaction = @view interactions[last_interaction, :]
        last_interaction = Int(interaction[4])
        energy += state[Int(interaction[1]), ibatch] ⊻ state[Int(interaction[2]), ibatch] ? -interaction[3] : interaction[3]
    end
    return energy
end

@inline function unsafe_energy_temperature(state::AbstractMatrix, temperature::AbstractMatrix,
                            interactions::AbstractMatrix, onsites::AbstractArray, last_interaction_index::AbstractArray,
                             inode::Integer, ibatch::Integer)
    last_interaction = last_interaction_index[inode]
    ret = state[inode, ibatch] == true ? onsites[inode] / temperature[inode, ibatch] : -onsites[inode] / temperature[inode, ibatch]
    while last_interaction != 0
        interaction = @view interactions[last_interaction, :]
        last_interaction = Int(interaction[4])
        energy = state[Int(interaction[1]), ibatch] ⊻ state[Int(interaction[2]), ibatch] ? -interaction[3] : interaction[3]
        ret += energy / sqrt(temperature[Int(interaction[1]), ibatch] * temperature[Int(interaction[2]), ibatch])
    end
    return ret
end

@inline function unsafe_step_kernel!(transition_rule::TransitionRule, temperature::AbstractMatrix{T},
    state::AbstractMatrix{Bool}, interactions::AbstractMatrix, onsites::AbstractArray, last_interaction_index::AbstractArray,
    ibatch::Integer, node::Integer) where T
    
    ΔE_over_T = -unsafe_energy_temperature(state, temperature, interactions, onsites, last_interaction_index, node, ibatch)
    state[node, ibatch] ⊻= true
    ΔE_over_T += unsafe_energy_temperature(state, temperature, interactions, onsites, last_interaction_index, node, ibatch)
    state[node, ibatch] ⊻= true

    if rand() < prob_accept(transition_rule, ΔE_over_T)
        state[node, ibatch] ⊻= true
    end
end

function update_temperature!(runtime::SpinSARuntime, temprule::ColumnWiseGradient, t::Integer, annealing_time::Integer, reverse_direction::Bool)
    dcut = cutoff_distance(temprule)
    sa = runtime.hamiltonian
    each_movement = (dcut * 2 + 2 * sa.m - 1) / (annealing_time - 1)
    middle_position = reverse_direction ? (2 * sa.m - 1) + dcut - t * each_movement : -dcut + t * each_movement
    temperature_matrix!(reshape(view(runtime.temperature, :, 1), sa.n, 2 * sa.m - 1), temprule, vcat(1:2:2*sa.m-1, 2:2:2*(sa.m-1)), middle_position)
    view(runtime.temperature, :, 2:size(runtime.temperature, 2)) .= view(runtime.temperature, :, 1:1)
end

function update_temperature!(runtime::SpinSARuntime, temp_scale::Vector{Float64}, t::Integer)
    view(runtime.temperature, :, 1) .= temp_scale[t]
    view(runtime.temperature, :, 2:size(runtime.temperature, 2)) .= view(runtime.temperature, :, 1:1)
end
    
_flip_match_device(::SpinSARuntime, spins) = spins


function track_equilibration_pulse!(
                runtime::SpinSARuntime,
                temprule::TemperatureGradient,
                annealing_time;
                tracker = nothing,
                flip_scheme = 1:nspin(runtime.model),
                transition_rule::TransitionRule = HeatBath(),
                reverse_direction::Bool=false
            )
    @assert all(sort!(vcat(flip_scheme...)) .== 1:nspin(runtime.model)) "invalid flip scheme: $flip_scheme, must be a perfect cover of all spins: $(1:nspin(runtime.model))"

    update_temperature!(runtime, temprule, 0, annealing_time, reverse_direction)
    for t in 1:annealing_time
        update_temperature!(runtime, temprule, t, annealing_time, reverse_direction)
        for spins in flip_scheme
            step!(runtime, transition_rule, _flip_match_device(runtime, spins))
        end
    end
end

function step!(runtime::SpinSARuntime, transition_rule::TransitionRule, simutanuous_flip_spins)
    for ibatch in 1:size(runtime.state, 2), spin in simutanuous_flip_spins
        unsafe_step_kernel!(transition_rule, runtime.temperature, runtime.state, runtime.model.interactions, runtime.model.onsites, runtime.model.last_interaction_index, ibatch, spin)
    end
end

function track_equilibration_plane!(
                runtime::SpinSARuntime,
                temp_scale::Vector{Float64},
                num_update_each_temp::Int;
                tracker = nothing,
                flip_scheme = 1:nspin(runtime.model),
                transition_rule::TransitionRule = HeatBath()
            )
    @show transition_rule
    for t in 1:length(temp_scale)
        @show t
        update_temperature!(runtime, temp_scale, t)
        for m in 1:num_update_each_temp
            # @show t, m
            for spins in flip_scheme
                step!(runtime, transition_rule, _flip_match_device(runtime, spins))
            end
        end
    end
end