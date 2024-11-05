abstract type TemperatureGradient end

# Gaussian temperature gradient: T(distance) = amplitude * exp(-(distance)^2 / width^2) + lowest_temperature
struct GaussianGradient <: TemperatureGradient
    amplitude::Float64
    width::Float64
    lowest_temperature::Float64
end
function evaluate_temperature(gg::GaussianGradient, distance::Real)
    return gg.amplitude * exp(-(distance/gg.width)^2) + gg.lowest_temperature
end
cutoff_distance(gg::GaussianGradient) = sqrt(-gg.width * log(1e-5/gg.amplitude))

# Exponential temperature gradient: T(distance) = amplitude * exp(-abs(distance) / width) + lowest_temperature
struct ExponentialGradient <: TemperatureGradient
    amplitude::Float64
    width::Float64
    lowest_temperature::Float64
end
function evaluate_temperature(eg::ExponentialGradient, distance::Real)
    return eg.amplitude * exp(-abs(distance/eg.width)) + eg.lowest_temperature
end
cutoff_distance(eg::ExponentialGradient) = -eg.width * log(1e-5/eg.amplitude)


abstract type TransitionRule end
struct HeatBath <: TransitionRule end
prob_accept(::HeatBath, ΔE_over_T::Real) = inv(1 + exp(ΔE_over_T))

struct Metropolis <: TransitionRule end
prob_accept(::Metropolis, ΔE_over_T::Real) = min(1, exp(-ΔE_over_T))