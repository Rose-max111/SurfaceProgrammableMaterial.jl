using SurfaceProgrammableMaterial
using Test

@testset "ca1d" begin
    include("ca1d.jl")
end

@testset "cusa" begin
    include("cusa.jl")
end

@testset "truthtablelp" begin
    include("truthtablelp.jl")
end
