module SurfaceProgrammableMaterial

using BitBasis
using CairoMakie
using CUDA
# using GenericTensorNetworks
using Random

# Export some abstract type used in code
export TempcomputeRule, Gaussiantype, Exponentialtype
export TransitionRule, HeatBath, Metropolis
export SimulatedAnnealingHamiltonian

# Export track_equilibration method
export track_equilibration_pulse_cpu!, track_equilibration_pulse_gpu!
export track_equilibration_pulse_reverse_cpu!, track_equilibration_pulse_reverse_gpu!
export track_equilibration_collective_temperature_cpu!, track_equilibration_collective_temperature_gpu!
export track_equilibration_fixedlayer_cpu!, track_equilibration_fixedlayer_gpu!

include("superstruct.jl")
include("ca1d.jl")
include("cusa.jl")

end
