module scoring
    import LinearAlgebra as La
    import Statistics as stats
    import Turing as tur

    export getSample
    export getSampleMatrix
    export getScore

    function getSample(chain::tur.MCMCChains.Chains,param::Symbol)
        return vec(chain[param].data)
    end

    function getSampleMatrix(chain::tur.MCMCChains.Chains,params::Vector{Symbol})
        # get samples nas flat arrays
        samples = [getSample(chain,p) for p in params]
    
        # build matrix from columns vectors
        return hcat(samples...)
    end

    function getScore(chain::tur.MCMCChains.Chains,params::Vector{Symbol},trueDict::Dict)
        # get sample matrix 
        mat = getSampleMatrix(chain,params)

        # get mean and covariance
        mean_vec = transpose(stats.mean(mat,dims=1))
        covmat = stats.cov(mat)

        # get true vector
        true_vec = [trueDict[p] for p in params]

        # get diff vector
        diff_vec = true_vec-mean_vec

        # compute score
        score = -log(La.det(covmat))-La.dot(diff_vec,inv(covmat)*diff_vec)

        return score
    end
end