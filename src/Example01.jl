#=
MDP:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-06
=#
include("common_defs.jl")
include("common_checks.jl")
include("valueiteration.jl")

const MinX = 0
const MaxX = 100
const StepX = 20

# mdp definition
mdp = MDP()

# state space
statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["no", "yes"])  # discrete

# action space
actionvariable!(mdp, "move", ["W", "E", "stop"])  # discrete

function isgoal(x::Float64)
  if abs(x - MaxX / 2) < StepX
    return "yes"
  else
    return "no"
  end
end

function mytransition(x::Float64, goal::AbstractString, move::AbstractString)
  if isgoal(x) == "yes" && goal == "yes"
    return [([x, isgoal(x)], 1.0)]
  end

  if move == "E"
    if x >= MaxX
      return [
        ([x, isgoal(x)], 0.9),
        ([x - StepX, isgoal(x - StepX)], 0.1)]
    elseif x <= MinX
      return [
        ([x, isgoal(x)], 0.2),
        ([x + StepX, isgoal(x + StepX)], 0.8)]
    else
      return [
        ([x, isgoal(x)], 0.1),
        ([x - StepX, isgoal(x - StepX)], 0.1),
        ([x + StepX, isgoal(x + StepX)], 0.8)]
    end
  elseif move == "W"
    if x >= MaxX
      return [
        ([x, isgoal(x)], 0.1),
        ([x - StepX, isgoal(x - StepX)], 0.9)]
    elseif x <= MinX
      return [
        ([x, isgoal(x)], 0.9),
        ([x + StepX, isgoal(x + StepX)], 0.1)]
    else
      return [
        ([x, isgoal(x)], 0.1),
        ([x - StepX, isgoal(x - StepX)], 0.8),
        ([x + StepX, isgoal(x + StepX)], 0.1)]
    end
  elseif move == "stop"
    return [([x, isgoal(x)], 1.0)]
  end
end

transition!(mdp, ["x", "goal", "move"], mytransition)

function myreward(x::Float64, goal::AbstractString, move::AbstractString)
  if goal == "yes" && move == "stop"
    return 1
  else
    return 0
  end
end

reward!(mdp, ["x", "goal", "move"], myreward)

solver = SerialValueIteration()

const StepX = 20
discretize_statevariable!(solver, "x", StepX)

lazyCheck(mdp, solver)

statespace, actionspace = getspaces(mdp, solver)
solver.stategrid = RectangleGrid(statespace...)
println(solver.stategrid)
solver.actiongrid = RectangleGrid(actionspace...)
println(solver.actiongrid)

println(length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap))

function solveset(mdp::MDP, svi::SerialValueIteration)

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(svi.stategrid)
  nactions = length(svi.actiongrid)

  vold = zeros(nstates)
  vnew = zeros(nstates)
  qval = zeros(nactions, nstates)
  resid = 0.0

  iter = 0
  itertime = 0.0
  cputime = 0.0

  for i in 1:svi.maxiter

    #tic()
    resid = -Inf

    for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = -Inf

      for iaction in 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)

        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)
        qnow = 0.0

        for istatep in 1:length(statepIdxs)
          qnow += probs[istatep] * vold[statepIdxs[istatep]]
        end

        qnow *= svi.discount
        qnow += reward(mdp, state, action)

        qval[iaction, istate] = qnow

        if qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end

      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? resid = newresid : nothing

    end

    #itertime = toq()
    #cputime += itertime

    if svi.verbose
      println(string("iter $iter, resid: $resid"))
    end

    resid < svi.tol ? break : nothing

    vtmp = vold
    vold = vnew
    vnew = vtmp
    iter = i

  end

  if iter == svi.maxiter
    @warn(string(
      "maximum number of iterations reached; check accuracy of solutions"))
  end

  @info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

  return ValueIterationSolution(
    qval,
    svi.stategrid,
    svi.actiongrid,
    cputime,
    iter,
    resid)
end

solveset(mdp, solver)
