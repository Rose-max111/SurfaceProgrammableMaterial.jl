abstract type TemperatureRule end
abstract type TemperatureGradient <: TemperatureRule end
abstract type ColumnWiseGradient <: TemperatureGradient end
abstract type TemperatureCollective <: TemperatureRule end

# Gaussian temperature gradient: T(distance) = amplitude * exp(-(distance)^2 / width^2) + lowest_temperature
struct GaussianGradient <: ColumnWiseGradient
    amplitude::Float64
    width::Float64
    lowest_temperature::Float64
end
function evaluate_temperature(gg::GaussianGradient, distance::Real)
    return gg.amplitude * exp(-(distance/gg.width)^2) + gg.lowest_temperature
end
cutoff_distance(gg::GaussianGradient) = sqrt(-gg.width * log(gg.lowest_temperature/gg.amplitude))
lowest_temperature(gg::GaussianGradient) = gg.lowest_temperature

# Exponential temperature gradient: T(distance) = amplitude * exp(-abs(distance) / width) + lowest_temperature
struct ExponentialGradient <: ColumnWiseGradient
    amplitude::Float64
    width::Float64
    lowest_temperature::Float64
end
function evaluate_temperature(eg::ExponentialGradient, distance::Real)
    return eg.amplitude * exp(-abs(distance/eg.width)) + eg.lowest_temperature
end
cutoff_distance(eg::ExponentialGradient) = -eg.width * log(eg.lowest_temperature/eg.amplitude)
lowest_temperature(eg::ExponentialGradient) = eg.lowest_temperature

# SigmoidGradient temperature gradient: T(distance) = low_temperature + (high_temperature - low_temperature) / (1 + exp(-distance / width))
# NOTE: I changed the order of the variables in the below struct definition
struct SigmoidGradient <: ColumnWiseGradient
    high_temperature::Float64
    width::Float64
    low_temperature::Float64
end
function evaluate_temperature(sg::SigmoidGradient, distance::Real)
    return sg.low_temperature + (sg.high_temperature - sg.low_temperature) / (1 + exp(-distance/sg.width))
end
cutoff_distance(sg::SigmoidGradient) = sg.width * log(sg.high_temperature/sg.low_temperature)
lowest_temperature(sg::SigmoidGradient) = sg.low_temperature

struct LinearTemperature <: TemperatureCollective
    high_temperature::Float64
    low_temperature::Float64
end
function evaluate_temperature(lt::LinearTemperature, proportion::Real) # proportion in [0, 1]
    return lt.high_temperature - proportion * (lt.high_temperature - lt.low_temperature)
end


abstract type TransitionRule end
struct HeatBath <: TransitionRule end
prob_accept(::HeatBath, ΔE_over_T::Real) = inv(1 + exp(ΔE_over_T))

struct Metropolis <: TransitionRule end
prob_accept(::Metropolis, ΔE_over_T::Real) = min(1, exp(-ΔE_over_T))