#=
valueiteration_serial:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-14
=#

function internalsolve(mdp::MDP, svi::SerialValueIteration)
  if length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap)
    return solveset(mdp, svi)
  elseif length(mdp.transition.argnames) > length(mdp.statemap) + length(mdp.actionmap)
    return solveprob(mdp, svi)
  else
    @error(string(
      "unkown transition function of neither T(s,a,s') nor T(s,a) type\n",
      "number of transition arguments: ",
      length(mdp.transition.argnames), "\n",
      "number of state and action variables",
      length(mdp.statemap) + length(mdp.actionmap)))
  end
end

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

function solveprob(mdp::MDP, svi::SerialValueIteration)

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
        qnow = 0.0

        for istatep in 1:nstates
          statep = getvar(svi.stategrid, mdp.statemap, stateargs, istatep)
          prob = mdp.transition.fn(state..., action..., statep...)
          qnow += prob * vold[istatep]
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
