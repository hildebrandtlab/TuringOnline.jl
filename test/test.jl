import Test as tst
import Random as rnd
import Distributions as dist
import Turing as tur
import AdvancedPS as aps
import DynamicPPL as ppl
rng = rnd.seed!(1)
include("../src/TuringOnline.jl")
using .online

tst.@testset "Parse symbols" begin
  tst.@test parseIndicesFromSymbol(Symbol("theta")) == (1,1) 
  tst.@test parseIndicesFromSymbol(Symbol("theta[3]")) == (3,1)
  tst.@test parseIndicesFromSymbol(Symbol("theta[3,3]")) == (3,3)
  tst.@test parseIndicesFromSymbol(Symbol("theta[3,33]")) == (3,33)
  tst.@test parseIndicesFromSymbol(Symbol("theta2[3,33]")) == (3,33)  
end

tst.@testset "Sort symbols in column major" begin
  syms = [Symbol("theta[2,1]"),  Symbol("theta[1,1]"),Symbol("theta[1,2]"),  Symbol("theta[2,2]")] 
  symsSorted = [Symbol("theta[1,1]"),Symbol("theta[2,1]"),Symbol("theta[1,2]"),Symbol("theta[2,2]")]    
  tst.@test sortSymbolsColumnMajor(syms) == symsSorted
    
  syms = [Symbol("theta[2]"),  Symbol("theta[1]"),Symbol("theta[3]"),  Symbol("theta[4]")] 
  symsSorted = [Symbol("theta[1]"),Symbol("theta[2]"),Symbol("theta[3]"),Symbol("theta[4]")]    
  tst.@test sortSymbolsColumnMajor(syms) == symsSorted
    
  syms = [Symbol("theta")] 
  symsSorted = [Symbol("theta")]    
  tst.@test sortSymbolsColumnMajor(syms) == symsSorted  
end

# We prepare data, model and a varinfo
data = [1 2 4 ; 3 5 7]

tur.@model testmodel(data::Matrix{Int64}) = begin
    # get dimensions
    (nc,nt)=size(data)
    
    # Hyperpriors
    alpha ~ dist.Exponential(1)
    beta ~ dist.Exponential(1)
    
    # latent space    
    theta ~ tur.filldist(dist.Beta(alpha,beta), nc,nt)
    
    # likelihood
    for i = 1:nc
        for j = 1:nt
            data[i,j] ~ dist.Binomial(10,theta[i,j])
        end    
    end
end;

model = testmodel(data)
varinfo = ppl.VarInfo(model)

tst.@testset "Writing metadata" begin
  # set new values 
  newVals = [1 2 3 4 5 6]
  chain = tur.MCMCChains.Chains(newVals, [:alpha, :beta, Symbol("theta[1,1]"),Symbol("theta[1,2]"),Symbol("theta[2,1]"),Symbol("theta[2,2]")])  
  varinfoNew = changeParticleCoordinates(varinfo,chain,chain,1,1)
  
  # check result
  tst.@test varinfoNew.metadata.alpha.vals == newVals[:,1]  
end

# We prepare an algorithm, sampler and a chain
algo = tur.SMC()
sampler = ppl.Sampler(algo)
nparticles = 12
chain = tur.sample(model, sampler, tur.MCMCThreads(), nparticles, 2)

tst.@testset "From chain to particles" begin
  # do conversion
  particles = convertChainToParticles(chain,model,sampler)
  
  # check results  
  alphaValsParticle1 = [particles[1].vals[i].f.varinfo.metadata.alpha.vals[1] for i in 1:nparticles]
  alphaValsChain1 = chain.value.data[:,1,1]   
  tst.@test alphaValsParticle1 == alphaValsChain1
    
  alphaValsParticle2 = [particles[2].vals[i].f.varinfo.metadata.alpha.vals[1] for i in 1:nparticles]
  alphaValsChain2 = chain.value.data[:,1,2]   
  tst.@test alphaValsParticle2 == alphaValsChain2
  
  betaValsParticle1 = [particles[1].vals[i].f.varinfo.metadata.beta.vals[1] for i in 1:nparticles]
  betaValsChain1 = chain.value.data[:,2,1]   
  tst.@test betaValsParticle1 == betaValsChain1
    
  betaValsParticle2 = [particles[2].vals[i].f.varinfo.metadata.beta.vals[1] for i in 1:nparticles]
  betaValsChain2 = chain.value.data[:,2,2]   
  tst.@test betaValsParticle2 == betaValsChain2
end

tst.@testset "From particles to chain" begin
  # do conversion
  particles = convertChainToParticles(chain,model,sampler)
    
  # do conversion back 
  chainBack = convertParticlesToChain(particles)
  
  # check results   
  tst.@test chain.value == chainBack.value
  
end
