abstract type TempcomputeRule end
struct Gaussiantype <: TempcomputeRule end
struct Exponentialtype <: TempcomputeRule end

abstract type TransitionRule end
struct HeatBath <: TransitionRule end
struct Metropolis <: TransitionRule end