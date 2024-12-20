# e.g. rule 110 could be specified as `CellularAutomata1D{110}()`
struct CellularAutomata1D{INT} end
CellularAutomata1D(id::Int) = CellularAutomata1D{id}()
nin(::CellularAutomata1D) = 3

function automatarule(p, q, r, N)
    return (N >> (p << 2 | q << 1 | r)) & 1
end
function (ca::CellularAutomata1D{N})(p, q, r) where N
    return automatarule(p, q, r, N)
end
rule110(p, q, r) = automatarule(p, q, r, 110)

# if not periodic_boundary, the boundary will be set to 0
function simulate(ca::CellularAutomata1D, initial_state::AbstractVector{Bool}, nstep::Int; periodic_boundary=true)
    n = length(initial_state)
    state = zeros(Bool, n, nstep+1)
    state[:, 1] .= initial_state
    for i in 2:nstep+1
        for j in 1:n
            a, b, c = if periodic_boundary
                state[mod1(j-1, n), i-1], state[j, i-1], state[mod1(j+1, n), i-1]
            else
                j == 1 ? 0 : state[j-1, i-1], state[j, i-1], j == n ? 0 : state[j+1, i-1]
            end
            state[j, i] = ca(a, b, c)
        end
    end
    return state
end
