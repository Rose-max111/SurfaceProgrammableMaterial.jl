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
    temperature::MT    # temperature, which has the same size as state
    state::ST    # state
    function SARuntime(hamiltonian::SimulatedAnnealingHamiltonian{ET}, temperature::AbstractMatrix{T}, state::AbstractMatrix{Bool}) where {T, ET}
        @assert size(temperature) == size(state) "temperature and state must have the same size"
        @assert nspin(hamiltonian) == size(state, 1) "state must have the same number of spins as the Hamiltonian"
        new{T, ET, typeof(temperature), typeof(state)}(hamiltonian, temperature, state)
    end
end
nbatch(sr::SARuntime) = size(sr.state, 2)
# initialize the runtime information with random state
function SARuntime(sa::SimulatedAnnealingHamiltonian{ET}, nbatch::Integer) where ET
    return SARuntime(sa, zeros(eltype(sa), nspin(sa), nbatch), zeros(eltype(sa), sa.m-1, nbatch), random_state(sa, nbatch))
end

@inline function unsafe_energy(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, inode::Integer, ibatch::Integer)
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        i, j = CartesianIndices((sa.n, sa.m))[inode].I
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        return trueoutput ⊻ state[inode, ibatch]
    end
end
@inline function unsafe_energy_over_temperature(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, temperature::AbstractMatrix, inode::Integer, ibatch::Integer)
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        i, j = CartesianIndices((sa.n, sa.m))[inode].I
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        Teff = (temperature[a, ibatch] * temperature[b, ibatch] * temperature[c, ibatch] * temperature[inode, ibatch])^0.25
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

# step on a subset of nodes and all batches
function step!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, temperature, nodes)
    for ibatch in 1:size(state, 2), node in nodes
        unsafe_step_kernel!(rule, sa, state, temperature, ibatch, node)
    end
end

@inline function unsafe_step_kernel!(rule::TransitionRule,
        sa::SimulatedAnnealingHamiltonian,
        state::AbstractMatrix,
        temperature::AbstractMatrix{T},
        ibatch::Integer,
        node::Integer
    ) where T
    @inbounds begin
        grid = CartesianIndices((sa.n, sa.m))
        i, j = grid[node].I  # proposed position

        ΔE_over_T_previous_layer = j > 1 ? ((ΔE, Teff) = unsafe_energy_over_temperature(sa, state, temperature, node, ibatch); (one(T) - 2 * ΔE) / Teff) : zero(T)
        ΔE_over_T_next_layer = zero(T)
        if j < sa.m # not the last layer
            # calculate the energy from the child nodes
            cnodes = unsafe_child_nodes(sa, node)
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_over_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer -= ΔE / Teff
            end
            # calculate the energy from the child nodes after flipping
            state[node, ibatch] ⊻= true
            for node in cnodes
                (ΔE, Teff) = unsafe_energy_over_temperature(sa, state, temperature, node, ibatch);
                ΔE_over_T_next_layer += ΔE / Teff
            end
            state[node, ibatch] ⊻= true
        end
        ΔE_over_T = ΔE_over_T_previous_layer + ΔE_over_T_next_layer
        if rand() < prob_accept(rule, ΔE_over_T)
            state[node, ibatch] ⊻= true
        end
    end
end
prob_accept(::HeatBath, ΔE_over_T::Real) = inv(1 + exp(ΔE_over_T))

function parallel_scheme(sa::SimulatedAnnealingHamiltonian{<:CellularAutomata1D})
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
        push!(ret, Vector(sa.n:2*sa.n:sa.n*sa.m))
        push!(ret, Vector(2*sa.n:2*sa.n:sa.n*sa.m))
    end
    if sa.n % 3 >= 2
        push!(ret, Vector(sa.n-1:2*sa.n:sa.n*sa.m))
        push!(ret, Vector(2*sa.n-1:2*sa.n:sa.n*sa.m))
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

function track_equilibration_pulse!(rule::TransitionRule,
                                        temprule::TemperatureGradient,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix, 
                                        annealing_time;
                                        tracker = nothing,
                                        flip_scheme = 1:nspin(sa)
                                        )
    @assert sort!(vcat(flip_scheme...)) == collect(1:nspin(sa)) "invalid flip scheme: $flip_scheme, must be a perfect cover of all spins: $(1:nspin(sa))"
    midposition = 1 - cutoff_distance(temprule)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    @info "midposition = $midposition"
    @info "each_movement = $each_movement"

    singlebatch_temp = temperature_matrix(temprule, sa, midposition)  # initial temperature
    temperature = repeat(vec(singlebatch_temp), 1, size(state, 2))
    tracker !== nothing && track!(tracker, copy(state), copy(temperature))
    for t in 1:annealing_time
        singlebatch_temp = temperature_matrix(temprule, sa, midposition + t * each_movement)
        temperature .= reshape(singlebatch_temp, :, 1)
        for spins in flip_scheme
            step!(rule, sa, state, temperature, spins)
        end
        tracker !== nothing && track!(tracker, copy(state), copy(temperature))
    end
end

function track_equilibration_collective_temperature!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,
                                        temperature,
                                        annealing_time; accelerate_flip = false)
    for t in 1:annealing_time
        singlebatch_temp = fill(temperature, sa.m-1)
        temperature = repeat(vec(singlebatch_temp), 1, size(state, 2))
        if accelerate_flip == false
            for thisspin in 1:nspin(sa)
                step!(rule, sa, state, temperature, thisspin)
            end
        else
            flip_list = parallel_scheme(sa)
            for eachflip in flip_list
                step!(rule, sa, state, temperature, eachflip)
            end
        end
    end
end

function track_equilibration_pulse_reverse!(rule::TransitionRule,
                                        temprule::TemperatureGradient,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix, 
                                        annealing_time;
                                        tracker = nothing,
                                        accelerate_flip = false
                                        )
    midposition = 1 - cutoff_distance(temprule)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    midposition = sa.m - 1.0 + 1.0 - midposition
    for t in 1:annealing_time
        @info "midposition = $midposition"
        singlebatch_temp = temperature_matrix(temprule, sa, midposition + (t-1) * each_movement)
        temperature = repeat(vec(singlebatch_temp), 1, size(state, 2))
        if accelerate_flip == false
            for thisspin in 1:nspin(sa)
                step!(rule, sa, state, temperature, thisspin)
            end
        else
            flip_list = parallel_scheme(sa)
            for eachflip in flip_list
                step!(rule, sa, state, temperature, eachflip)
            end
        end
    end
end
function temperature_matrix(tg::TemperatureGradient, sa::SimulatedAnnealingHamiltonian, middle_position::Real)
    return [evaluate_temperature(tg, middle_position - j) for i in 1:sa.n, j in 1:sa.m]
end

function track_equilibration_fixedlayer!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,  
                                        annealing_time;
                                        accelerate_flip = false,
                                        fixedinput = false)
    Temp_list = reverse(range(1e-5, 10.0, annealing_time))
    for this_temp in Temp_list
        singlebatch_temp = fill(this_temp, sa.m-1)
        temperature = repeat(vec(singlebatch_temp), 1, size(state, 2))
        if accelerate_flip == false
            if fixedinput == false # this time fix output
                for thisspin in 1:nspin(sa)-sa.n
                    step!(rule, sa, state, temperature, thisspin)
                end
            else # this time fix input
                for thisspin in sa.n+1:nspin(sa)
                    step!(rule, sa, state, temperature, thisspin)
                end
            end
        else
            flip_list = parallel_scheme(SimulatedAnnealingHamiltonian(sa.n, sa.m-1)) # original fix output
            if fixedinput == true # this time fix input
                flip_list = [[x + sa.n for x in inner_vec] for inner_vec in flip_list]
            end
            for eachflip in flip_list
                step!(rule, sa, state, temperature, eachflip)
            end
        end
    end
end

