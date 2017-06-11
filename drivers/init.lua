require 'paths'

local drivers = {}

function drivers.init(model, workers, opts)
  local state = {}

  state.gpuImgs = torch.CudaTensor(opts.batchSize, 3,opts.imsize,opts.imsize)
  state.gpuInstrs = torch.CudaTensor(opts.maxSeqlen, opts.batchSize, opts.stDim)
  state.gpuIngrs = torch.CudaTensor(opts.batchSize, opts.maxIngrs)
  state.gpuMatches = torch.CudaTensor(opts.batchSize)
  state.gpuClsRecipe = torch.CudaTensor(opts.batchSize)
  state.gpuClsImg = torch.CudaTensor(opts.batchSize)

  cosine_crit = nn.CosineEmbeddingCriterion(0.1):cuda()
  weights_class = torch.Tensor(opts.numClasses):fill(1)
  weights_class[1] = 0 -- class background has 0 weight

  if opts.semantic then
    class_crit = nn.ClassNLLCriterion(weights_class):cuda()
    -- first table output is given to cosine_crit, other 2 outputs are handled with NLL
    state.clsw = opts.clsw
    state.cosw = opts.cosw

    parallel_crit = nn.ParallelCriterion():add(cosine_crit,state.cosw):add(class_crit,state.clsw):add(class_crit,state.clsw)
    parallel_rec_only = nn.ParallelCriterion():add(cosine_crit,0):add(class_crit,0):add(class_crit,state.clsw)
    parallel_im_only = nn.ParallelCriterion():add(cosine_crit,0):add(class_crit,state.clsw):add(class_crit,0)

    -- keep original criterions to display loss
    state.crit = {cosine_crit,class_crit,parallel_crit}
    --state.crit = {cosine_crit,class_crit,parallel_crit,parallel_rec_only,parallel_im_only}
  else
    state.crit = cosine_crit
  end

  state.t = 0
  state.valtrack = 0
  state.dswap = 1
  local drivers = {nil, nil, nil}
  local lazyDrivers = {nil, nil, nil}

  for i,driver in pairs({'train', 'val', 'snap', 'test'}) do
    drivers[i] = function(...) lazyDrivers[i](...) end

    lazyDrivers[i] = function(...)
      lazyDrivers[i] = paths.dofile(driver..'.lua')(model, workers, opts, state)
      lazyDrivers[i](...)
    end
  end

  return table.unpack(drivers)
end

return drivers
