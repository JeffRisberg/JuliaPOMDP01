#=
MDP:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-06-01
=#
# using Pkg; Pkg.add("POMDPs"); Pkg.add("QMDP")
# using Pkg; Pkg.add("QuickPOMDPs"); Pkg.add("POMDPModelTools"); Pkg.add("POMDPSimulators"); Pkg.add("POMDPPolicies")

using POMDPs, QMDP, QuickPOMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies

struct MyMDP <: MDP{Int64, Int64} # MDP{StateType, ActionType}
end

struct GridWorldState
    x::Int64 # x position
    y::Int64 # y position
    done::Bool # are we in a terminal state?
end

# initial state constructor
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)
# checks if the position of two states are the same
posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y

# the grid world mdp type
mutable struct GridWorld <: MDP{GridWorldState, Symbol} # Note that our MDP is parametarized by the state and the action
    size_x::Int64 # x size of the grid
    size_y::Int64 # y size of the grid
    reward_states::Vector{GridWorldState} # the states in which agent recieves reward
    reward_values::Vector{Float64} # reward values for those states
    tprob::Float64 # probability of transitioning to the desired state
    discount_factor::Float64 # disocunt factor
end

#we use keyworded arguments so we can change any of the values we pass in
function GridWorld(;sx::Int64=4, # size_x
                    sy::Int64=3, # size_y
                    rs::Vector{GridWorldState}=[GridWorldState(4,2), GridWorldState(4,3)], # reward states
                    rv::Vector{Float64}=rv = [-1.0, 1.0], # reward values
                    tp::Float64=0.8, # tprob
                    discount_factor::Float64=0.9)
    return GridWorld(sx, sy, rs, rv, tp, discount_factor)
end

# we can now create a GridWorld mdp instance like this:
mdp = GridWorld()
mdp.reward_states # mdp contains all the defualt values from the constructor

function POMDPs.states(mdp::GridWorld)
    s = GridWorldState[] # initialize an array of GridWorldStates
    # loop over all our states, remeber there are two binary variables:
    # done (d)
    for d = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
        push!(s, GridWorldState(x,y,d))
    end
    return s
end;

POMDPs.actions(mdp::GridWorld) = [:up, :down, :left, :right];

# transition helpers
function inbounds(mdp::GridWorld,x::Int64,y::Int64)
    if x == 2 && y == 2
        return false
    elseif 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
        return true
    else
        return false
    end
end

inbounds(mdp::GridWorld, state::GridWorldState) = inbounds(mdp, state.x, state.y);

function POMDPs.transition(mdp::GridWorld, state::GridWorldState, action::Symbol)
    a = action
    x = state.x
    y = state.y

    if state.done
        return SparseCat([GridWorldState(x, y, true)], [1.0])
    elseif state in mdp.reward_states
        return SparseCat([GridWorldState(x, y, true)], [1.0])
    end

    neighbors = [
        GridWorldState(x+1, y, false), # right
        GridWorldState(x-1, y, false), # left
        GridWorldState(x, y-1, false), # down
        GridWorldState(x, y+1, false), # up
        ] # See Performance Note below

    targets = Dict(:right=>1, :left=>2, :down=>3, :up=>4) # See Performance Note below
    target = targets[a]

    probability = fill(0.0, 4)

    if !inbounds(mdp, neighbors[target])
        # If would transition out of bounds, stay in
        # same cell with probability 1
        return SparseCat([GridWorldState(x, y)], [1.0])
    else
        probability[target] = mdp.tprob

        oob_count = sum(!inbounds(mdp, n) for n in neighbors) # number of out of bounds neighbors

        new_probability = (1.0 - mdp.tprob)/(3-oob_count)

        for i = 1:4 # do not include neighbor 5
            if inbounds(mdp, neighbors[i]) && i != target
                probability[i] = new_probability
            end
        end
    end

    return SparseCat(neighbors, probability)
end

function POMDPs.reward(mdp::GridWorld, state::GridWorldState, action::Symbol, statep::GridWorldState) #deleted action
    if state.done
        return 0.0
    end
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end;

POMDPs.discount(mdp::GridWorld) = mdp.discount_factor;

function POMDPs.stateindex(mdp::GridWorld, state::GridWorldState)
    sd = Int(state.done + 1)
    ci = CartesianIndices((mdp.size_x, mdp.size_y, 2))
    return LinearIndices(ci)[state.x, state.y, sd]
end

function POMDPs.actionindex(mdp::GridWorld, act::Symbol)
    if act==:up
        return 1
    elseif act==:down
        return 2
    elseif act==:left
        return 3
    elseif act==:right
        return 4
    end
    error("Invalid GridWorld action: $act")
end;

POMDPs.isterminal(mdp::GridWorld, s::GridWorldState) = s.done

POMDPs.initialstate(pomdp::GridWorld) = Deterministic(GridWorldState(1,1)) # TODO: define initialistate for states, not distributions?

# using Pkg; Pkg.add("DiscreteValueIteration")

# first let's load the value iteration module
using DiscreteValueIteration

# initialize the problem
mdp = GridWorld()

# initialize the solver
# max_iterations: maximum number of iterations value iteration runs for (default is 100)
# belres: the value of Bellman residual used in the solver (default is 1e-3)
solver = ValueIterationSolver(max_iterations=10, belres=1e-7; verbose=true)

# solve for an optimal policy
policy = solve(solver, mdp)
println(policy)

# say we are in state (1,1)
s = GridWorldState(1,1)
println(action(policy, s))
println(value(policy, s))

# say we are in state (2,1)
s = GridWorldState(2,1)
println(action(policy, s))
println(value(policy, s))

# say we are in state (3,1)
s = GridWorldState(3,1)
println(action(policy, s))
println(value(policy, s))

iob = IOBuffer()
io = IOContext(iob, :limit=>true, :displaysize=>(80, 80))
show(io, MIME("text/plain"), policy)
println(String(take!(iob)))
