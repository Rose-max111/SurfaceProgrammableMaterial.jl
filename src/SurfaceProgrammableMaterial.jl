module SurfaceProgrammableMaterial

using BitBasis
using Random
using JuMP
using GenericTensorNetworks
using COPT
using HiGHS

# Temperature gradient
export GaussianGradient, ExponentialGradient, SigmoidGradient, ColumnWiseGradient, TemperatureGradient, StationaryExponentialGradient, StationaryColumnWiseGradient
export StepExponentialGradient

# Temperature collective
export LinearTemperature, TemperatureCollective

# Transition rule
export TransitionRule, HeatBath, Metropolis
export SimulatedAnnealingHamiltonian, energy, random_state

# Basic gate
export BasicGate

# Cellular automata transition rule
export CellularAutomata1D, automatarule

# Track equilibration method
export track_equilibration_pulse!, SAStateTracker, SARuntime, why
export parallel_scheme

# Truth table mapping in LP method
export query_model, nspin, ground_state, IsingGadget

# Visualize method
export show_temperature_matrix, animate_tracker, show_temperature_curve, show_effective_temperature_curve

# Spin model
export IsingModel, spin_model_construction, SpinSARuntime, track_equilibration_plane!
export spin_to_bool, bool_to_spin

include("superstruct.jl")
include("basicgate.jl")
include("ca1d.jl")
include("simulated_annealing.jl")
include("truthtablelp.jl")
include("visualize.jl")
include("spin_model.jl")

end
