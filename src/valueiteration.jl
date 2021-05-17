using GridInterpolations

export
  SerialValueIteration,
  ParallelValueIteration,
  discretize_statevariable!

const MaxIter = 1000
const Tol = 1e-4
const Discount = 0.99
const NThreads = 1

# a change in value for a variable
mutable struct LazyDiscrete
  varname::AbstractString
  step::Float64

  LazyDiscrete(varname::AbstractString, step::Float64) = new(varname, step)
end

abstract type ValueIteration <: Solver end

# these are the parameters of a Solver
mutable struct SerialValueIteration <: ValueIteration
  verbose::Bool

  maxiter::Int64
  tol::Float64
  discount::Float64

  statemap::Dict{AbstractString, LazyDiscrete}
  actionmap::Dict{AbstractString, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  SerialValueIteration(;
      verbose::Bool=true,
      maxiter::Real=MaxIter,
      tol::Float64=Tol,
      discount::Real=Discount) =
    new(
      verbose,
      Int64(maxiter),
      tol,
      Float64(discount),
      Dict{AbstractString, LazyDiscrete}(),
      Dict{AbstractString, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())
end

# these are the parameters of a Solver
mutable struct ParallelValueIteration <: ValueIteration
  nthreads::Int64
  verbose::Bool

  maxiter::Int64
  tol::Float64
  discount::Float64

  statemap::Dict{AbstractString, LazyDiscrete}
  actionmap::Dict{AbstractString, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  ParallelValueIteration(;
      nthreads::Real=NThreads,
      verbose::Bool=true,
      maxiter::Real=MaxIter,
      tol::Float64=Tol,
      discount::Real=Discount) =
    new(
      Int64(nthreads),
      verbose,
      Int64(maxiter),
      tol,
      Float64(discount),
      Dict{AbstractString, LazyDiscrete}(),
      Dict{AbstractString, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())
end

mutable struct ValueIterationSolution <: Solution
  qval::Matrix{Float64}  # nactions x nstates Q-value matrix
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  cputime::Float64
  niter::Int64
  finaltol::Float64

  ValueIterationSolution(
      qval::Matrix{Float64},
      stategrid::RectangleGrid,
      actiongrid::RectangleGrid,
      cputime::Float64,
      niter::Int64,
      finaltol::Float64) =
    new(qval, stategrid, actiongrid, cputime, niter, finaltol)
end

function discretize_statevariable!(vi::ValueIteration, varname::AbstractString, step::Real)
  if haskey(vi.statemap, varname)
    @warn(string(
      "state variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.statemap[varname] = LazyDiscrete(varname, Float64(step))
end

function discretize_actionvariable!(vi::ValueIteration, varname::AbstractString, step::Real)
  if haskey(vi.actionmap, varname)
    @warn(string(
      "action variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.actionmap[varname] = LazyDiscrete(varname, Float64(step))
end

# extract a policy from a solution
function getpolicy(mdp::MDP, solution::ValueIterationSolution)
  statedim = length(mdp.statemap)
  actiondim = length(mdp.actionmap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]
  nactions, nstates = size(solution.qval)

  function indexify(statevec::Vector)
    stateidxvec = zeros(statedim)
    for idim in 1:statedim
      statevar = mdp.statemap[stateargs[idim]]
      if isa(statevar, RangeVar)
        stateidxvec[idim] = statevec[idim]
      elseif isa(statevar, ValuesVar)
        stateidxvec[idim] = findfirst(==(statevar.values), statevec[idim])
      else
        error(string(
          "unknown state variable definition type for ", statevar))
      end
    end
    return stateidxvec
  end

  function policy(state...)
    statevec = [stateelem for stateelem in state]
    stateidxvec = indexify(statevec)
    stateidxs, wts = interpolants(solution.stategrid, stateidxvec)

    iaction_best = 0
    vaction_best = -Inf

    for iaction in 1:nactions
      vaction = 0.0
      for i in 1:length(stateidxs)
        vaction += wts[i] * solution.qval[iaction, stateidxs[i]]
      end

      if vaction > vaction_best
        iaction_best = iaction
        vaction_best = vaction
      end
    end

    rawaction = ind2x(solution.actiongrid, iaction_best)
    action = Array{Any,1}(undef, actiondim)

    for i in 1:actiondim
      actionvar = mdp.actionmap[actionargs[i]]
      if isa(actionvar, RangeVar)
        action[i] = rawaction[i]
      elseif isa(actionvar, ValuesVar)
        action[i] = actionvar.values[Int64(rawaction[i])]
      else
        error(string(
          "unknown action variable definition type for ", actionvar))
      end
    end

    return action
  end

  return policy
end

include("valueiteration_checks.jl")
include("valueiteration_solver.jl")
