using SurfaceProgrammableMaterial
using Test

@testset "ca1d" begin
    include("ca1d.jl")
end

@testset "simulated annealing" begin
    include("simulated_annealing.jl")
end

@testset "truthtablelp" begin
    include("truthtablelp.jl")
end
