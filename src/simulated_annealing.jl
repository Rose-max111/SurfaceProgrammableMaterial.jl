struct SimulatedAnnealingHamiltonian{ET}
    n::Int # number of spins per layer
    m::Int # number of full layers (count the last layer!)
    energy_term::ET  # e.g. a cellular automaton rule, takes 3 inputs and returns 1 output
    periodic::Bool  # whether to use periodic boundary condition
end
SimulatedAnnealingHamiltonian(n::Int, m::Int, energy_term) = SimulatedAnnealingHamiltonian(n, m, energy_term, true)

nspin(sa::SimulatedAnnealingHamiltonian) = sa.n * sa.m
spins(sa::SimulatedAnnealingHamiltonian) = Base.OneTo(nspin(sa))
function random_state(sa::SimulatedAnnealingHamiltonian, nbatch::Integer)
    return rand(Bool, nspin(sa), nbatch)
end

# NOTE: MT and ST can be CuArray type.
struct SARuntime{T, MT<:AbstractMatrix{T}, ST<:AbstractMatrix{Bool}, ET}    # runtime information for simulated annealing
    hamiltonian::SimulatedAnnealingHamiltonian{ET}   # the Hamiltonian
    energy_gradient::MT    # energy gradient
    temperature::MT    # temperature, which has the same size as state
    state::ST    # state
    function SARuntime(hamiltonian::SimulatedAnnealingHamiltonian{ET}, energy_gradient::AbstractArray{T}, temperature::AbstractMatrix{T}, state::AbstractMatrix{Bool}) where ET
        @assert size(energy_gradient) == size(temperature) == size(state) "energy_gradient, temperature and state must have the same size"
        @assert nspin(hamiltonian) == size(state, 1) "state must have the same number of spins as the Hamiltonian"
        new{T, typeof(energy_gradient), typeof(state), ET}(hamiltonian, energy_gradient, temperature, state)
    end
end
nbatch(sr::SARuntime) = size(sr.state, 2)

# initialize the runtime information with random state
function SARuntime(sa::SimulatedAnnealingHamiltonian{ET}, nbatch::Integer) where ET
    return SARuntime(sa, zeros(eltype(sa), nspin(sa), nbatch), zeros(eltype(sa), sa.m-1, nbatch), rand(Bool, nspin(sa), nbatch))
end

# evaluate the energy of the i-th gadget (involving spins i and its parents)
function calculate_energy(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, ibatch::Integer)
    return sum(i->unsafe_evaluate_parent(sa, state, i, ibatch), sa.n+1:nspin(sa))
end
@inline function unsafe_evaluate_parent(sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, inode::Integer, ibatch::Integer)
    (a, b, c) = unsafe_parent_nodes(sa, inode)
    @inbounds begin
        i, j = CartesianIndices((sa.n, sa.m))[inode].I
        trueoutput = sa.energy_term(state[a, ibatch], state[b, ibatch], state[c, ibatch])
        return trueoutput ⊻ state[inode, ibatch]
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

# step on single node and all batches
function step!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, temperature, node=nothing)
    @boundscheck size(temperature) == size(state) || throw(DimensionMismatch("temperature must have the same size as state"))
    for ibatch in 1:size(state, 2)
        unsafe_step_kernel!(rule, sa, state, temperature, ibatch, node)
    end
    state
end

@inline function unsafe_step_kernel!(rule::TransitionRule,
        sa::SimulatedAnnealingHamiltonian,
        state::AbstractMatrix,
        temperature::AbstractMatrix,
        ibatch::Integer,
        node::Integer
    )
    ΔE_with_next_layer = 0
    ΔE_with_previous_layer = 0
    grid = CartesianIndices((sa.n, sa.m))

    @inbounds begin
        i, j = grid[node].I
        if j > 1 # not the first layer
            ΔE_with_previous_layer += 1 - 2 * unsafe_evaluate_parent(sa, state, node, ibatch)
        end
        if j < sa.m # not the last layer
            cnodes = unsafe_child_nodes(sa, node)
            for node in cnodes
                ΔE_with_next_layer -= unsafe_evaluate_parent(sa, state, node, ibatch)
            end
            # flip the node@
            state[node, ibatch] ⊻= true
            for node in cnodes
                ΔE_with_next_layer += unsafe_evaluate_parent(sa, state, node, ibatch)
            end
            state[node, ibatch] ⊻= true
        end
        flip_max_prob = 1
        if j == sa.m
            flip_max_prob *= prob_accept(rule, temperature[ibatch][j-1], ΔE_with_previous_layer)
        elseif j == 1
            flip_max_prob *= prob_accept(rule, temperature[ibatch][j], ΔE_with_next_layer)
        else
            flip_max_prob = 1.0 / (1.0 + exp(ΔE_with_previous_layer / temperature[ibatch][j-1] + ΔE_with_next_layer / temperature[ibatch][j]))
        end
        if rand() < flip_max_prob
            state[node, ibatch] ⊻= true
            ΔE_with_next_layer
        else
            0
        end
    end
end
prob_accept(::HeatBath, temperature, ΔE::Real) = inv(1 + exp(ΔE / temperature))

function step_parallel!(rule::TransitionRule, sa::SimulatedAnnealingHamiltonian, state::AbstractMatrix, temperature, flip_id)
    # @info "flip_id = $flip_id"
    for ibatch in 1:size(state, 2)
        for this_time_flip in flip_id
            unsafe_step_kernel!(rule, sa, state, temperature, ibatch, this_time_flip)
        end
    end
    state
end

function toymodel_pulse(rule::TempcomputeRule, sa::SimulatedAnnealingHamiltonian,
                            pulse_amplitude::Float64,
                            pulse_width::Float64,
                            middle_position::Float64,
                            gradient::Float64)
    # amplitude * e^(- (1 /width) * (x-middle_position)^2)
    # eachposition = Tuple([pulse_amplitude * gradient^(- (1.0/pulse_width) * abs(i-middle_position)) + 1e-5 for i in 1:sa.m-1])
    eachposition = [temp_calculate(rule, pulse_amplitude, pulse_width, middle_position, gradient, i) for i in 1:sa.m-1]
    return eachposition 
end
temp_calculate(::Gaussiantype, pulse_amplitude::Float64,
                            pulse_width::Float64,
                            middle_position::Float64,
                            gradient::Float64,
                            i) = pulse_amplitude * gradient^(- (1.0/pulse_width) * (i-middle_position)^2) + 1e-5
temp_calculate(::Exponentialtype, pulse_amplitude::Float64,
                            pulse_width::Float64,
                            middle_position::Float64,
                            gradient::Float64,
                            i) = pulse_amplitude * gradient^(- (1.0/pulse_width) * abs(i-middle_position)) + 1e-5

function get_parallel_flip_id(sa)
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

midposition_calculate(::Exponentialtype, 
                    pulse_amplitude::Float64,
                    pulse_width::Float64,
                    energy_gradient::Float64) = 1.0 - (-(pulse_width) * log(1e-5/pulse_amplitude) / log(energy_gradient))
midposition_calculate(::Gaussiantype, 
                    pulse_amplitude::Float64,
                    pulse_width::Float64,
                    energy_gradient::Float64) = 1.0 - sqrt(-(pulse_width) * log(1e-5/pulse_amplitude) /log(energy_gradient))

function track_equilibration_pulse_cpu!(rule::TransitionRule,
                                        temprule::TempcomputeRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix, 
                                        pulse_gradient, 
                                        pulse_amplitude::Float64,
                                        pulse_width::Float64,
                                        annealing_time; accelerate_flip = false
                                        )
    midposition = midposition_calculate(temprule, pulse_amplitude, pulse_width, pulse_gradient)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    @info "midposition = $midposition"
    @info "each_movement = $each_movement"

    for t in 1:annealing_time
        singlebatch_temp = toymodel_pulse(temprule, sa, pulse_amplitude, pulse_width, midposition, pulse_gradient)
        temperature = repeat(singlebatch_temp, 1, size(state, 2))
        energy_gradient = fill(1.0, size(state, 2))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, energy_gradient, temperature, thisatom)
            end
        else
            flip_list = get_parallel_flip_id(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, energy_gradient, temperature, eachflip)
            end
        end
        midposition += each_movement
    end
end

function track_equilibration_collective_temperature_cpu!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,
                                        temperature,
                                        annealing_time; accelerate_flip = false)
    for t in 1:annealing_time
        singlebatch_temp = fill(temperature, sa.m-1)
        temperature = repeat(singlebatch_temp, 1, size(state, 2))
        energy_gradient = fill(1.0, size(state, 2))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, energy_gradient, temperature, thisatom)
            end
        else
            flip_list = get_parallel_flip_id(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, energy_gradient, temperature, eachflip)
            end
        end
    end
end

function track_equilibration_pulse_reverse_cpu!(rule::TransitionRule,
                                        temprule::TempcomputeRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix, 
                                        pulse_gradient, 
                                        pulse_amplitude::Float64,
                                        pulse_width::Float64,
                                        annealing_time;
                                        accelerate_flip = false
                                        )
    midposition = midposition_calculate(temprule, pulse_amplitude, pulse_width, pulse_gradient)
    each_movement = ((1.0 - midposition) * 2 + (sa.m - 2)) / (annealing_time - 1)
    midposition = sa.m - 1.0 + 1.0 - midposition

    single_layer_temp = []
    for t in 1:annealing_time
        @info "midposition = $midposition"
        singlebatch_temp = toymodel_pulse(temprule, sa, pulse_amplitude, pulse_width, midposition, pulse_gradient)
        temperature = repeat(singlebatch_temp, 1, size(state, 2))
        if accelerate_flip == false
            for thisatom in 1:nspin(sa)
                step!(rule, sa, state, temperature, thisatom)
            end
        else
            flip_list = get_parallel_flip_id(sa)
            for eachflip in flip_list
                step_parallel!(rule, sa, state, energy_gradient, temperature, eachflip)
            end
        end
        midposition -= each_movement
        # push!(single_layer_temp, singlebatch_temp[1])
    end
    # return single_layer_temp
end

function track_equilibration_fixedlayer_cpu!(rule::TransitionRule,
                                        sa::SimulatedAnnealingHamiltonian, 
                                        state::AbstractMatrix,  
                                        annealing_time; accelerate_flip = false, fixedinput = false)
    Temp_list = reverse(range(1e-5, 10.0, annealing_time))
    for this_temp in Temp_list
        singlebatch_temp = fill(this_temp, sa.m-1)
        temperature = repeat(singlebatch_temp, 1, size(state, 2))
        energy_gradient = fill(1.0, size(state, 2))
        if accelerate_flip == false
            if fixedinput == false # this time fix output
                for thisatom in 1:nspin(sa)-sa.n
                    step!(rule, sa, state, energy_gradient, temperature, thisatom)
                end
            else # this time fix input
                for thisatom in sa.n+1:nspin(sa)
                    step!(rule, sa, state, energy_gradient, temperature, thisatom)
                end
            end
        else
            flip_list = get_parallel_flip_id(SimulatedAnnealingHamiltonian(sa.n, sa.m-1)) # original fix output
            if fixedinput == true # this time fix input
                flip_list = [[x + sa.n for x in inner_vec] for inner_vec in flip_list]
            end
            for eachflip in flip_list
                step_parallel!(rule, sa, state, energy_gradient, temperature, eachflip)
            end
        end
    end
end

