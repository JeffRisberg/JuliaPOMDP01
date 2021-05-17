function lazyCheck(mdp::MDP, vi::ValueIteration)
  checkArgumentOrder(mdp) # common_checks
  checkTransition(mdp) # common_checks
  checkDiscretize(mdp, vi)
  @info(string("mdp and value iteration solver passed basic checks"))
end

# Check that all |mdp| RangeVar variables have valid discretization schemes in |vi|
function checkDiscretize(mdp::MDP, vi::ValueIteration)

  for lazyvar in values(mdp.statemap)
    if isa(lazyvar, RangeVar)
      if !haskey(vi.statemap, lazyvar.varname)
        @error(string(
          "state variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < vi.statemap[lazyvar.varname].step
        @error(string(
          "state variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end

  for lazyvar in values(mdp.actionmap)
    if isa(lazyvar, RangeVar)
      if !haskey(vi.actionmap, lazyvar.varname)
        @error(string(
          "action variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < vi.actionmap[lazyvar.varname].step
        @error(string(
          "action variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end
end
