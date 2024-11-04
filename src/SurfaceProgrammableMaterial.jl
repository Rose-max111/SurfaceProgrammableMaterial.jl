module SurfaceProgrammableMaterial

using BitBasis
using CairoMakie
using Random
using JuMP
using GenericTensorNetworks
using COPT
using HiGHS
using Suppressor


# Export some abstract type used in code
export TempcomputeRule, Gaussiantype, Exponentialtype
export TransitionRule, HeatBath, Metropolis
export SimulatedAnnealingHamiltonian

# Export cellular automata transition ruleid
export CellularAutomata1D
export automatarule

# Export track_equilibration method
export track_equilibration_pulse_cpu!, track_equilibration_pulse_gpu!
export track_equilibration_pulse_reverse_cpu!, track_equilibration_pulse_reverse_gpu!
export track_equilibration_collective_temperature_cpu!, track_equilibration_collective_temperature_gpu!
export track_equilibration_fixedlayer_cpu!, track_equilibration_fixedlayer_gpu!

# Export truth_table_mapping in LP method
export query_model, nspin, ground_state, IsingGadget

include("superstruct.jl")
include("ca1d.jl")
include("simulated_annealing.jl")
include("truthtablelp.jl")

end
