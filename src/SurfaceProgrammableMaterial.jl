module SurfaceProgrammableMaterial

using BitBasis
using Random
using JuMP
using GenericTensorNetworks
using COPT
using HiGHS
using Suppressor


# Export some abstract type used in code
export GaussianGradient, ExponentialGradient, TemperatureGradient
export TransitionRule, HeatBath, Metropolis
export SimulatedAnnealingHamiltonian, energy, random_state

# Basic gate
export BasicGate

# Export cellular automata transition ruleid
export CellularAutomata1D
export automatarule

# Export track_equilibration method
export track_equilibration_pulse!, track_equilibration_pulse_reverse!, SAStateTracker
export track_equilibration_collective_temperature!, track_equilibration_fixedlayer!

# Export truth_table_mapping in LP method
export query_model, nspin, ground_state, IsingGadget

export show_temperature_matrix, animate_tracker

include("superstruct.jl")
include("basicgate.jl")
include("ca1d.jl")
include("simulated_annealing.jl")
include("truthtablelp.jl")
include("visualize.jl")

end
