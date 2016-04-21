using DataFrames
# using StatsBase
using GLM
using Distributions
using NLopt
include("pglmm_gaussian.jl")

pwd()
veg = readtable("veg.csv") # the vegetation data: site, sp, abundance
Vphy = readtable("phy.csv") # the phylogenetic var-cov matrix
nspp = length(unique(veg[:sp]))
nsite = length(unique(veg[:site]))
# set up random effects
re_sp = Array[[1], veg[:sp], eye(nspp)];
re_sp_phy = Array[[1], veg[:sp], Vphy];
re_site = Array[[1], veg[:site], eye(nsite)];
re_nested_phy = Array[[1], veg[:sp], Vphy, veg[:site]];
re = Array[re_sp, re_sp_phy, re_nested_phy, re_site];
# to track number of iterations
global feval
feval = 0
communityPGLMM_gaussian(Y ~ 1, veg, re)
@time test = communityPGLMM_gaussian(Y ~ 1, veg, re)
test["ss"] # estimated random effects.
