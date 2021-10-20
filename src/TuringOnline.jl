module online
  import Setfield as sfl
  import DynamicPPL as ppl
  import Turing as tur
  import AdvancedPS as aps
  import Random as rnd

  export changeParticleCoordinates  
  export convertChainToParticles  
  export convertParticlesToChain
  export update
  export getParameterFromMetadata
  export parseIndicesFromSymbol
  export sortSymbolsColumnMajor
  """
      parseIndicesFromSymbol(s::Symbol)

  Parse symbols "varname", "varname[i]" and "varname[i,j]" to return (1,1),(i,1) and (i,j).
  """ 
  function parseIndicesFromSymbol(s::Symbol)
    # convert to string
    str = string(s)
    
    # check for case "varname[i,j]"
    m = match(r"(\S+)\[(\d+),(\d+)\]", str)
    if !isnothing(m)
        # return (i,j)
        return (parse(Int, m[2]),parse(Int, m[3]))
    end
    
    # check for case "varname[i]"
    m = match(r"(\S+)\[(\d+)\]", str)
    if !isnothing(m)
        # return (i,1)
        return (parse(Int, m[2]),1)
    end 
    
    # else we assume "varname"
    return (1,1)
  end

  """
      sortSymbolsColumnMajor(symbols::Vector{Symbol})

  Sorts a vector of symbols of the one of the types "varname", "varname[i]" or "varname[i,j]" at a time in column-major order.
  """ 
  function sortSymbolsColumnMajor(symbols::Vector{Symbol})
   # add indices to list
   indices = [ parseIndicesFromSymbol(s) for s in symbols] 
   
   # sort zipped lists
   zipped = [(i,s) for (i,s) in zip(indices,symbols)]     
   sorted = sort(zipped,lt=((ix,sx),(iy,sy))-> ix[2] < iy[2] || (ix[2] == iy[2] && ix[1] < iy[1]))
    
   # return only symbols
   return [s for (i,s) in sorted]
  end

  """
          changeParticleCoordinates(varinfo::ppl.VarInfo{<:NamedTuple},chainPoster::tur.MCMCChains.Chains,chainPrior,indexParticle::Int64,indexChain::Int64)

  Reads particle coordinates from chainPoster or chainPrior and sets them in the varinfo.
  """
  function changeParticleCoordinates(varinfo::ppl.VarInfo{<:NamedTuple},chainPoster::tur.MCMCChains.Chains,chainPrior,indexParticle::Int64,indexChain::Int64)
        # get higher level metadata
        metaHigh = varinfo.metadata

        for key in keys(metaHigh) 
            # get metadata of params        
            meta = metaHigh[key]    

            # get parameter name
            name = meta.vns[1]
            
            # get parameter range in varinfo
            range = meta.ranges[1]
            numberFields = size(range,1)
        
            # get symbols from chain
            #symbolsChainPoster = sort([s for s in fieldnames(typeof(tur.MCMCChains.get(chainPoster,key,flatten=true)))])
            symbolsChainPoster = [s for s in fieldnames(typeof(tur.MCMCChains.get(chainPoster,key,flatten=true)))]
                    
            # get parameter ranges in chain
            numberFieldsChainPoster = length(symbolsChainPoster)
                    
            # write in what we have from the chains
            particleCoord = if !isnothing(chainPrior)
                # get prio symbols                
                symbolsChainPrior = sortSymbolsColumnMajor([s for s in fieldnames(typeof(tur.MCMCChains.get(chainPrior,key,flatten=true)))])
                # if symbol in poster, use this, else use from prior
            [if (s in symbolsChainPoster) chainPoster.value[var=s].data[indexParticle,indexChain] else chainPrior.value[var=s].data[indexParticle,1] end for s in symbolsChainPrior]
            else
                # get all symbols from poster chain
                [chainPoster.value[var=s].data[indexParticle,indexChain] for s in symbolsChainPoster]
            end
           
        
            # set value in varinfo
            varinfo = sfl.@set varinfo.metadata[key].vals = particleCoord
            #ppl.setval!(varinfo, particleCoord, name)

       end 

       return varinfo 
    end

  """
      convertChainToParticles(chain::tur.MCMCChains.Chains,model::ppl.Model,sampler::ppl.Sampler)

  Converts a MCMC chain into a vector of particle containers using the parameter samples from the chain as particle coordinates.
  """    
  function convertChainToParticles(chain::tur.MCMCChains.Chains,model::ppl.Model,sampler::ppl.Sampler)
    # Get parameters in chain
    paramsChain = Set(chain.name_map.parameters)
            
    # get varinfo from model
    varinfo = ppl.VarInfo(model)
    
    # Get parameters from model
    paramsModel = Set(getParameterFromMetadata(varinfo.metadata))
    
    # check if they are equal
    diff = setdiff(paramsModel,paramsChain)
    newParameters = length(diff) > 0
    if newParameters
        println("Some parameters in are not present in the chain but in the model: $diff \nUsing prior samples from the model for them instead.")        
    end
    
    # read values from chain
    nparticles = size(chain.value.data,1)
    
    # get number of chains    
    nchains = size(chain.value,3)    
    
    # check if the chain has weights
    hasWeights = :weight in chain.name_map.internals
    
    weights = if hasWeights
            chain[:weight].data
        else 
            fill(1.0,nparticles,nchains)
        end
    
    # get prior samples if necessary
    priorChain = if newParameters prior_chain = tur.sample(model, tur.Prior(), nparticles) else nothing end
    
    # prepare storage for traces
    dictTraces = Dict()   
    for j in 1:nchains
      dictTraces[j] = Vector{aps.Trace}()
      sizehint!(dictTraces[j], nparticles)
    end
    
    # for each chain and particle...
    Threads.@threads for j in 1:nchains    
            for i in 1:nparticles
 
            # get new varinfo
            newVarInfo = changeParticleCoordinates(varinfo,chain,priorChain,i,j)
                
            # get trace
            trace = aps.Trace(model,sampler,newVarInfo)
        
            # append
            push!(dictTraces[j],trace)                
        end
    end        
    return [aps.ParticleContainer(dictTraces[j],weights[:,j]) for j in 1:nchains]
  end

  """
      convertParticlesToChain(particles::Vector{aps.ParticleContainer{aps.Trace}})

  Converts a vector of particle containers into a MCMC chain using the particle coordinates as parameter samples in the chain.
  """     
  function convertParticlesToChain(particles::Vector{aps.ParticleContainer{aps.Trace}})
    # get number of chains
    nchains = length(particles)
        
    # get number of particles
    nparticles = size(particles[1].vals,1)
    
    # check that particles are not empty
    @assert(nparticles > 0)
    
    # get a metadata object
    meta = particles[1].vals[1].f.varinfo.metadata
        
    nCoordinates = sum([length(particles[1].vals[1].f.varinfo.metadata[key].vals) for key in keys(meta)]) +2    
    
    # prepare storage for samples
    storageType = Float64
    samples = zeros(nparticles,nCoordinates,nchains)
      
    # extract particle coordinates, lp and weights
    Threads.@threads for j in 1:nchains    
      for i in 1:nparticles
       # prepare storage
       coordinates = Vector{storageType}()
        
       # add particle coordinates 
       for key in keys(meta)
            coord = particles[j].vals[i].f.varinfo.metadata[key].vals
            for c in coord
                push!(coordinates,c)
            end
       end
       
       # add lp
       #push!(coordinates,particles.vals[i].f.varinfo.logp[])
       push!(coordinates,0.0)
        
       # add weight
       push!(coordinates,particles[j].logWs[i])
       
        
       # write to storage  
       #push!(dictSamples[j],coordinates) 
       samples[i,:,j] = coordinates         
      end
    end
      # get parameter names  
      pnames_symbol = getParameterFromMetadata(meta)
    
      # add lp and weights
      push!(pnames_symbol,"lp")
      push!(pnames_symbol,"weight")
    
      # construct chain
      chain = tur.MCMCChains.Chains(samples,pnames_symbol,Dict(:internals => [:lp,:weight]))
    
      return chain
    
  end

  """
      getParameterFromMetadata(meta)

  Extracts symbols for model parameters from metadata.
  """
  function getParameterFromMetadata(meta)
    # get shapes for parameters
    shapes = [ size(meta[key].dists[1]) for key in keys(meta) ]
    
    # get parameter ranges    
    numberFields =  [if length(s) == 2 s[1]*s[2] elseif length(s) == 1 s[1] else 1 end  for s in shapes]
        
    # get parameter names  
    pnames = [p for p in fieldnames(typeof(meta))]
    
    # write symbols    
    pnames_symbol = Vector()
    for (index,name) in enumerate(pnames)
        # scalars    
        if numberFields[index] == 1
            push!(pnames_symbol,name)
        # vectors        
        elseif length(shapes[index]) == 1
            for i in 1:numberFields[index]
                push!(pnames_symbol,Symbol(name,"[$i]"))
            end
        # matrices        
        else
            (cols,rows) = shapes[index]    
            for i in 1:cols
                for j in 1:rows    
                    push!(pnames_symbol,Symbol(name,"[$i,$j]"))
                end        
            end 
        end    
    end
    return pnames_symbol
  end

  """
      update(chain::tur.MCMCChains.Chains,model_new_data::ppl.Model,algo::T,rng::rnd.AbstractRNG) where {T<:tur.Inference.ParticleInference}

  Updates parameter samples in chain given the model. If the model introduces parameters that are not present in the chain, prior samples are used as initial values for the updating.
  """
  function update(chain::tur.MCMCChains.Chains,
        model_new_data::ppl.Model,
        algo::T,
        rng::rnd.AbstractRNG) where {T<:tur.Inference.ParticleInference}
    # create sampler
    sampler = ppl.Sampler(algo);
    
    # create particles and convert to chain
    particles = convertChainToParticles(chain,model_new_data,sampler)
    #oldChain = convertParticlesToChain(particles)
    
    # update particles and convert to chain
    Threads.@threads for j in 1:length(particles)       
      logevidence = aps.sweep!(rng, particles[j], sampler.alg.resampler)
    end
    updatedChain = convertParticlesToChain(particles)
    
    return updatedChain#,oldChain
  end

end