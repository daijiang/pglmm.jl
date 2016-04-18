using DataFrames
# using StatsBase
Pkg.add("GLM")
using GLM
using Distributions
using Optim

pwd()
veg = readtable("veg.csv")
Vphy = readtable("phy.csv")

B_init = Nullable()
s2_init = Nullable()
REML = true
verbose = true

nspp =  length(unique(veg[:sp]))
nsite =  length(unique(veg[:site]))
sp = veg[:sp]
site = veg[:site]

mf = ModelFrame(Y ~ 1, veg);
X = ModelMatrix(mf).m
Y = convert(Vector{Float64},DataFrames.model_response(mf))
re_sp = Array[[1], veg[:sp], eye(nspp)]
re_sp_phy = Array[[1], veg[:sp], Vphy]
re_site = Array[[1], veg[:site], eye(nsite)]
re_nested_phy = Array[[1], veg[:sp], Vphy, veg[:site]]
re = Array[re_sp, re_sp_phy, re_nested_phy, re_site]
q = length(re)
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

if isnull(B_init)
  B_init = coef(lm(Y~1, veg))
end

if isnull(s2_init)
  s2_init = var(residuals(lm(Y~1, veg))) / q
end

B = B_init
s = repmat([sqrt(s2_init)], q)
hcat(s)
# begin pglmm.LL here
# include("pglmm_LL.jl")
function pglmm_LL(par) #; X = X, Y = Y, Zt = Zt, St = St, nested = nested, REML = REML, verbose = verbose
  n, p = size(X)
  # par = s
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
    println((vcat(LL, par))')
  end

  return(LL)
end

###############################################################
optimize(pglmm_LL, s)


###############################################################
