function checkArgumentOrder(mdp::MDP)
  statedim = length(mdp.statemap)
  actiondim = length(mdp.actionmap)

  ntransitionargs = length(mdp.transition.argnames)
  nrewardargs = length(mdp.reward.argnames)

  sadim = statedim + actiondim
  saspdim = sadim + statedim

  if sadim != nrewardargs
    @error(string(
      "the number of reward function input arguments must be the same as ",
      "the sum of the number of state and action variables"))
  elseif sadim != ntransitionargs && saspdim != ntransitionargs
    @error(string(
      "the number of transition function input arguments must be the same as ",
      "the sum of the number of state, action, and next state variables or ",
      "the sum of the number of state and action variables"))
  end

  for iarg in 1:nrewardargs
    if mdp.reward.argnames[iarg] != mdp.transition.argnames[iarg]
      @error(string(
        "transition and reward function state and action variable input arguments",
        "must be consistent in both naming and order"))
    end
  end

  if length(mdp.transition.argnames) > length(mdp.reward.argnames)
    for iarg in 1:nrewardargs - actiondim
      if mdp.transition.argnames[iarg] != mdp.transition.argnames[iarg + sadim]
        @error(string(
          "transition type T(s,a,s')'s state s and next state s' variable ",
          "input arguments must be consistent in both naming and order"))
      end
    end
  end
end

function checkTransition(mdp::MDP)
  args = randargs(mdp)
  transitionval = mdp.transition.fn(args...)

  if isa(transitionval, Real) || isa(transitionval, Vector)
    checkTransition(mdp, args, transitionval)
  else
    @error(string(
      "transition function provided is not a correctly defined ",
      "T(s,a,s') or T(s,a) type function, check the return type\n",
      "return type: ", typeof(transitionval)))
  end
end

function checkTransition(mdp::MDP, args::Vector, transitionval::Real)
  if transitionval < 0 || transitionval > 1
    @warn(string(  # warn not error because we might have sampled a non-existent state
      "transition function provided is of type T(s,a,s'), ",
      "but the value returned from a random state is not ",
      "a valid probability value bounded to [0,1]\n",
      "argument names: ", mdp.transition.argnames, "\n",
      "random state: ", args, "\n",
      "return value: ", transitionval))
  end
end

function checkTransition(mdp::MDP, args::Vector, transitionval::Vector)
  nargs = length(mdp.transition.argnames)
  sumprob = 0.0

  for returnval in transitionval  # |returnval| is a (state, prob) pair
    state = returnval[1]
    prob = returnval[2]
    sumprob += prob

    if length(state) + length(prob) != nargs || !isvalid(mdp, state)
      @error(string(
        "transition function provided is of type T(s,a), ",
        "but one of the states returned from a random state is ",
        "either not bounded by its range or not in the set of values\n",
        "argument names: ", mdp.transition.argnames, "\n",
        "random state: ", args, "\n",
        "return value: ", state, "\n",
        "probability: ", prob))
    elseif prob < 0 || prob > 1
      @error(string(
        "transition function provided is of type T(s,a), ",
        "but one of the probabilities returned from a random state is ",
        "a valid probability value bounded to [0,1]\n",
        "argument names: ", mdp.transition.argnames, "\n",
        "random state: ", args, "\n",
        "return value: ", state, "\n",
        "probability: ", prob))
    end
  end

  if sumprob != 1
    @warn(string(  # warn not error because we might have sampled a non-existent state
      "transition function provided is of type T(s,a), ",
      "but the sum of transition probabilities returned from a random state ",
      "does not sum to 1\n",
      "argument names: ", mdp.transition.argnames, "\n",
      "random state: ", args, "\n",
      "return value: ", returnval, "\n",
      "sum of transition probabilities: ", sumprob))
  end
end

function randargs(mdp::MDP)
  nargs = length(mdp.transition.argnames)
  args = Array{Any,1}(undef, nargs)
  for iarg in 1:nargs
    argname = mdp.transition.argnames[iarg]
    if haskey(mdp.statemap, argname)
      args[iarg] = lazySample(mdp.statemap[argname])
    elseif haskey(mdp.actionmap, argname)
      args[iarg] = lazySample(mdp.actionmap[argname])
    else
      @error(string("variable ", lazyvar.varname, " has not been defined"))
    end
  end
  args
end

function lazySample(lazyvar::LazyVar)
  if isa(lazyvar, RangeVar)
    return lazyvar.minval + rand() * (lazyvar.maxval - lazyvar.minval)
  elseif isa(lazyvar, ValuesVar)
    return lazyvar.values[rand(1:length(lazyvar.values))]
  else
    @error(string("variable", lazyvar.varname, " is not a valid subtype of LazyVar"))
  end
end

function isvalid(mdp::MDP, state::Vector)
  for iarg in length(mdp.statemap)
    argname = mdp.transition.argnames[iarg]
    if isa(mdp.statemap[argname], RangeVar) &&
        (state[iarg] < mdp.statemap[argname].minval ||
        state[iarg] > mdp.statemap[argname].maxval)
      return false
    elseif isa(mdp.statemap[argname], ValuesVar) &&
        !(state[iarg] in mdp.statemap[argname].values)
      return false
    end
  end
  return true
end
