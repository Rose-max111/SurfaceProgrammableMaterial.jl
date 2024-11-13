using DelimitedFiles, Test, BenchmarkTools, Statistics

"""General Annealing Problem"""
abstract type AnnealingProblem end

"""
    SpinAnnealingProblem{T<:Real} <: AnnealingProblem

Annealing problem defined by coupling matrix of spins.
"""
struct SpinAnnealingProblem{T<:Real} <: AnnealingProblem  # immutable, with type parameter T (a subtype of Real).
    num_spin::Int
    coupling::Matrix{T}
    function SpinAnnealingProblem(coupling::Matrix{T}) where T
        size(coupling, 1) == size(coupling, 2) || throw(DimensionMismatch("input must be square matrix."))
        new{T}(size(coupling, 1), coupling)
    end
end

"""
    load_coupling(filename::String) -> SpinAnnealingProblem

Load the data file into symmtric coupling matrix.
"""
function load_coupling(filename::String)
    data = readdlm(filename)
    is = Int.(view(data, :, 1)) .+ 1  #! @. means broadcast for the following functions, is here used correctly?
    js = Int.(view(data, :, 2)) .+ 1
    weights = data[:,3]
    num_spin = max(maximum(is), maximum(js))
    J = zeros(eltype(weights), num_spin, num_spin)
    @inbounds for (i, j, weight) = zip(is, js, weights)
        J[i,j] = weight/2
        J[j,i] = weight/2
    end
    SpinAnnealingProblem(J)
end

# @testset "loading" begin
#     sap = load_coupling("test/externel_std/example.txt")
#     for i in 1:sap.num_spin
#         @test sap.coupling[i, i] == 0 
#     end
#     @test size(sap.coupling) == (300, 300)
# end

abstract type AnnealingConfig end

struct SpinConfig{Ts, Tf} <: AnnealingConfig
    config::Vector{Ts}
    field::Vector{Tf}
end

"""
    random_config(prblm::AnnealingProblem) -> SpinConfig

Random spin configuration.
"""
function random_config end   # where to put the docstring of a multiple-dispatch function is a problem. Using `abstract function` is proper.

function random_config(prblm::SpinAnnealingProblem)
    config = rand([-1,1], prblm.num_spin)
    SpinConfig(config, prblm.coupling*config)
end

# @testset "random config" begin
#     sap = load_coupling("test/externel_std/example.txt")
#     initial_config = random_config(sap)
#     @test initial_config.config |> length == 300
#     @test eltype(initial_config.config) == Int
# end

"""
    anneal_singlerun!(config::AnnealingConfig, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * configuration that can be updated.
    * prblm: problem with `get_cost`, `flip!` and `random_config` interfaces.
    * tempscales: temperature scales, which should be a decreasing array.
    * num_update_each_temp: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)
    cost = get_cost(config, prblm)
    
    opt_config = config
    opt_cost = cost
    for beta = 1 ./ tempscales
        @simd for m = 1:num_update_each_temp  # single instriuction multiple data, see julia performance tips.
            proposal, ΔE = propose(config, prblm)
            if exp(-beta*ΔE) > rand()  #accept
                flip!(config, proposal, prblm)
                cost += ΔE
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = config
                end
            end
        end
    end
    opt_cost, opt_config
end
 
"""
    anneal(nrun::Int, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing with multiple runs.
"""
function anneal(nrun::Int, prblm, tempscales::Vector{Float64}, num_update_each_temp::Int)
    local opt_config, opt_cost
    for r = 1:nrun
        initial_config = random_config(prblm)
        cost, config = anneal_singlerun!(initial_config, prblm, tempscales, num_update_each_temp)
        if r == 1 || cost < opt_cost
            opt_cost = cost
            opt_config = config
        end
    end
    opt_cost, opt_config
end

"""
    get_cost(config::AnnealingConfig, ap::AnnealingProblem) -> Real

Get the cost of specific configuration.
"""
get_cost(config::SpinConfig, sap::SpinAnnealingProblem) = sum(config.config'*sap.coupling*config.config)

"""
    propose(config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(config::SpinConfig, ::SpinAnnealingProblem)  # ommit the name of argument, since not used.
    ispin = rand(1:length(config.config))
    @inbounds ΔE = -config.field[ispin] * config.config[ispin] * 4 # 2 for spin change, 2 for mutual energy.
    ispin, ΔE
end

"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
@inline function flip!(config::SpinConfig, ispin::Int, sap::SpinAnnealingProblem)
    @inbounds config.config[ispin] = -config.config[ispin]  # @inbounds can remove boundary check, and improve performance
    @simd for i=1:sap.num_spin
        @inbounds config.field[i] += 2 * config.config[ispin] * sap.coupling[i,ispin]
    end
    config
end

using Random
Random.seed!(2)
const tempscales = 10 .- (1:64 .- 1) .* 0.15 |> collect
const sap = load_coupling("100_example.txt")

# @testset "anneal" begin
#     opt_cost, opt_config = anneal(30, sap, tempscales, 2000)
#     @show opt_cost
#     # @test anneal(30, sap, tempscales, 4000)[1] == -3858
#     # anneal(30, sap, tempscales, 4000)
#     # res = median(@benchmark anneal(30, $sap, $tempscales, 4000))
#     # @test res.time/1e9 < 2
#     # @test res.allocs < 500
# end

opt_cost, opt_config = anneal(30, sap, tempscales, 50)
@show opt_cost

# @benchmark anneal(30, $sap, $tempscales, 4000)