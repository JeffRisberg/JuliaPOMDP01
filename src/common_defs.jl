#=
common_defs:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-14
=#
abstract type LazyVar end

mutable struct RangeVar <: LazyVar
  varname::AbstractString
  minval::Float64
  maxval::Float64

  function RangeVar(varname::AbstractString, minval::Float64, maxval::Float64)
    if minval > maxval
      throw(ArgumentError("minimum value must be smaller than maximum value"))
    end
    new(varname, minval, maxval)
  end
end

mutable struct ValuesVar <: LazyVar
  varname::AbstractString
  values::Vector

  ValuesVar(varname::AbstractString, values::Vector) = new(varname, values)
end

mutable struct LazyFunc
  argnames::Vector{String}
  fn::Function

  LazyFunc() = new(AbstractString[], function emptyfunc() end)
  LazyFunc(argnames::Vector{String}, fn::Function) = new(argnames, fn)
end

mutable struct MDP
  statemap::Dict{AbstractString, LazyVar}
  actionmap::Dict{AbstractString, LazyVar}
  transition::LazyFunc
  reward::LazyFunc

  MDP() = new(
    Dict{AbstractString, LazyVar}(),
    Dict{AbstractString, LazyVar}(),
    LazyFunc(),
    LazyFunc())
end

abstract type Solver end
abstract type Solution end

function statevariable!(mdp::MDP, varname::AbstractString, minval::Real, maxval::Real)
  if haskey(mdp.statemap, varname)
    @warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = RangeVar(varname, Float64(minval), Float64(maxval))
end

function statevariable!(mdp::MDP, varname::AbstractString, values::Vector)
  if haskey(mdp.statemap, varname)
    @warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = ValuesVar(varname, values)
end

function actionvariable!(mdp::MDP, varname::AbstractString, minval::Real, maxval::Real)
  if haskey(mdp.actionmap, varname)
    @warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = RangeVar(varname, Float64(minval), Float64(maxval))
end

function actionvariable!(mdp::MDP, varname::AbstractString, values::Vector)
  if haskey(mdp.actionmap, varname)
    @warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = ValuesVar(varname, values)
end

# |argnames| is an ordered list of argument names for |transition|
function transition!(mdp::MDP, argnames::Vector{String}, transition::Function)
  mdp.transition = LazyFunc(argnames, transition)
end

# |argnames| is an ordered list of argument names for |reward|
function reward!(mdp::MDP, argnames::Vector{String}, reward::Function)
  mdp.reward = LazyFunc(argnames, reward)
end

function solve(mdp::MDP, solver::Solver)
  lazyCheck(mdp, solver)
  return lazySolve(mdp, solver)
end
