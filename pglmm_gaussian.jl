function communityPGLMM_gaussian(
  formula::Formula,
  data::DataFrame,
  random_effect::Vector;
  sp = nothing,
  site = nothing,
  B_init = nothing,
  s2_init = nothing,
  REML::Bool = false,
  verbose::Bool = false,
  reltol::Real = 1e-8,
  maxit::Integer = 1000
  )
  if sp == nothing
    sp = data[:sp]
  end
  if site == nothing
    site = data[:site]
  end

  nspp =  length(unique(sp))
  nsite =  length(unique(site))

  mf = ModelFrame(formula, veg);
  X = ModelMatrix(mf).m
  Y = convert(Vector{Float64}, DataFrames.model_response(mf))

  q = length(random_effect)
  re = copy(random_effect)
  St_lengths = fill(0, q)
  Ztt = Array[]
  nested = Array[]
  ii = 0
  jj = 0
  i = 1
  for i in 1:q
    re_i = re[i]
    # non-nested terms
    if length(re_i) == 3
      counter = 0
      Z_i = zeros(nspp * nsite, length(unique(re_i[2])))
      for i_levels in unique(re_i[2])
        counter = counter + 1
        Z_i[:, counter] = re_i[1] .* (i_levels .== re_i[2])
      end
      Zt_i = chol(re_i[3]) * (Z_i')
      ii = ii + 1
      push!(Ztt, Zt_i)
      St_lengths[ii] = length(unique(re_i[2]))
    end

    if length(re_i) == 4
      if all(re_i[2] .== sp)
        nestedsp_j = re_i[3]
        nestedsite_j = eye(nsite)
        nested_j = kron(nestedsite_j, nestedsp_j)
      end
      if all(re_i[2] .== site)
        nestedsp_j = eye(nspp)
        nestedsite_j = re_i[3]
        nested_j = kron(nestedsite_j, nestedsp_j)
      end
      jj = jj + 1
      push!(nested, nested_j)
    end
  end

  q_nonNested = ii
  q_Nested = jj

  if q_nonNested > 0
    St = zeros(q_nonNested, sum(St_lengths))
    Zt = zeros(sum(St_lengths), nspp*nsite)
    count = 1
    for i in 1:q_nonNested
      St[i, count:(count + St_lengths[i] - 1)] = ones(1, St_lengths[i])
      Zt[count:(count + St_lengths[i] - 1), :] = Ztt[i]
      count = count + St_lengths[i]
    end
  else
    St = Array[]
    Zt = Array[]
  end
  # sparse(St)
  n, p = size(X) ### attention here !!!!

  if B_init == nothing
    B_init = coef(lm(Y~1, veg))
  end

  if s2_init == nothing
    s2_init = var(residuals(lm(Y~1, veg))) / q
  end

  B = B_init
  s = repmat([sqrt(s2_init)], q)
  # par = s

  # begin pglmm.LL here
  feval::Int = 0
  global feval # to track number of iterations
  function pglmm_LL(par::Vector, grad::Vector)
    # grad not used, but needed for NLopt.Opt
    global feval
    feval::Int += 1
    n, p = size(X)
    if !isempty(St)
      q_nonNested = size(St, 1)
      sr = real(par[1:q_nonNested])
      iC = sr[1] * St[1, :]
      if length(sr) > 1
        for i in 2:q_nonNested
          iC = iC + sr[i] * St[i, :]
        end
      end
      iC = diagm(squeeze(iC, 1))
      Ut = iC * Zt
      U = Ut'
    else
      q_nonNested = 0
      sr = Array[]
    end

    if isempty(nested)
      q_Nested = 0
    else
      q_Nested = length(nested)
    end

    if q_Nested == 0
      sn = Nullable()
    else
      sn = real(par[(q_nonNested + 1):(q_nonNested + q_Nested)])
    end

    if q_Nested == 0
      iA = eye(n)
      Ishort = eye(size(Ut, 1))
      Ut_iA_U = Ut * iA * U
      iV = iA - U * inv(Ishort + Ut_iA_U) * Ut
    else
      A = eye(n)
      for j in 1:q_Nested
        A = A + sn[j]^2 * nested[j]
      end
      iA = inv(A)
      if q_nonNested > 0
        Ishort = eye(size(Ut, 1))
        Ut_iA_U = Ut * iA * U
        iV = iA - iA * U * inv(Ishort + Ut_iA_U) * Ut * iA
      else
        iV = iA
      end
    end

    denom = X' * iV * X
    num = X' * iV * Y
    B = \(denom, num)
    H = Y - X * B

    if q_Nested == 0
      logdetV = logdet(Ishort + Ut_iA_U)
      if isinf(logdetV)
        logdetV = 2 * sum(log(diag(chol(Ishort + Ut_iA_U))))
      end
    else
      logdetV = -logdet(iV)
      if isinf(logdetV)
        logdetV = -2 * sum(log(diag(cholfact(iV, :U, Val{true})[:U])))
      end
      if isinf(logdetV)
        logdetV = 10^10
      end
    end

    if REML == true
      s2_conc = (H' * iV * H) / (n - p)
      LL = 0.5 * ((n - p) * log(s2_conc) + logdetV + (n - p) + logdet(denom))
    else
      s2_conc = (H' * iV * H) / n
      LL = 0.5 * (n * log(s2_conc) + logdetV + n)
    end

    if verbose == true
      println((vcat(feval, LL, par))')
    end

    return(LL[1,1])
  end

  if q > 1
    opt = NLopt.Opt(:LN_NELDERMEAD, length(s))
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, reltol)   # criterion on parameter value changes
    NLopt.min_objective!(opt, pglmm_LL)
    LL, par, ret = NLopt.optimize(opt, s)
    # opt = optimize(pglmm_LL, s, method = :nelder_mead,
    #              grtol = reltol,
    #              iterations = maxit)
    else
      opt = NLopt.Opt(:LD_LBFGS, length(s))
      NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
      NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
      NLopt.xtol_abs!(opt, reltol)   # criterion on parameter value changes
      NLopt.min_objective!(opt, pglmm_LL)
      LL, par, ret = NLopt.optimize(opt, s)
      # opt = optimize(pglmm_LL, s, method = :l_bfgs,
      #               grtol = reltol, iterations = maxit)
  end

  par = abs(par)
  niter = feval
  convcode = ret

  ###############################################################
  if !isempty(St)
    q_nonNested = size(St, 1)
    sr = real(par[1:q_nonNested])
    iC = sr[1] * St[1, :]
    if length(sr) > 1
      for i in 2:q_nonNested
        iC = iC + sr[i] * St[i, :]
      end
    end
    iC = diagm(squeeze(iC, 1))
    Ut = iC * Zt
    U = Ut'
  else
    q_nonNested = 0
    sr = Array[]
  end

  if isempty(nested)
    q_Nested = 0
  else
    q_Nested = length(nested)
  end

  if q_Nested == 0
    sn = Nullable()
  else
    sn = real(par[(q_nonNested + 1):(q_nonNested + q_Nested)])
  end

  if q_Nested == 0
    iA = eye(n)
    Ishort = eye(size(Ut, 1))
    Ut_iA_U = Ut * iA * U
    iV = iA - U * inv(Ishort + Ut_iA_U) * Ut
  else
    A = eye(n)
    for j in 1:q_Nested
      A = A + sn[j]^2 * nested[j]
    end
    iA = inv(A)
    if q_nonNested > 0
      Ishort = eye(size(Ut, 1))
      Ut_iA_U = Ut * iA * U
      iV = iA - iA * U * inv(Ishort + Ut_iA_U) * Ut * iA
    else
      iV = iA
    end
  end

  denom = X' * iV * X
  num = X' * iV * Y
  B = \(denom, num)
  H = Y - X * B

  if q_Nested == 0
    logdetV = logdet(Ishort + Ut_iA_U)
    if isinf(logdetV)
      logdetV = 2 * sum(log(diag(chol(Ishort + Ut_iA_U))))
    end
  else
    logdetV = -logdet(iV)
    if isinf(logdetV)
      logdetV = -2 * sum(log(diag(cholfact(iV, :U, Val{true})[:U])))
    end
    if isinf(logdetV)
      logdetV = 10^10
    end
  end

  if REML == true
    s2resid = (H' * iV * H) / (n - p)
  else
    s2resid = (H' * iV * H) / n
  end

  s2r = s2resid .* (sr.^2)
  s2n = s2resid .* (sn.^2)
  ss = vcat(sr, sn, s2resid.^0.5)

  iV = iV ./ s2resid

  B_cov = inv(X' * iV * X)
  B_se = diag(B_cov).^0.5
  B_zscore = B/B_se
  B_pvalue =  2 * (1 - cdf(Normal(0, 1.0), abs(B_zscore)))

  if REML == true
    loglik = -0.5 * (n - p) * log(2 * pi) + 0.5 * logdet(X'X) - LL[1]
  else
    loglik = -0.5 * n * log(2 * pi) - LL[1]
  end

  k = p + q + 1
  aic = -2 * loglik + 2 * k
  bic = -2 * loglik + k * (log(n) - log(pi))

  Dict("B" => B,
  "B_se" => B_se,
  "B_cov" => B_cov,
  "B_zscore" => B_zscore,
  "B_pvalue" => B_pvalue,
  "ss" => ss,
  "s2r" => s2r,
  "s2n" => s2n,
  "s2resid" => s2resid,
  "loglik" => loglik,
  "aic" => aic,
  "bic" => bic,
  "s2_init" => s2_init,
  "B_init" => B_init,
  "Y" => Y,
  "X" => X,
  "H" => H,
  "iV" => iV,
  "convcode" => convcode,
  "niter" => niter)
end
