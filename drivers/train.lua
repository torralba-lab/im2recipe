require 'optim'
require 'math'
return function(model, workers, opts, state)

  local gpuImgs = state.gpuImgs
  local gpuInstrs = state.gpuInstrs
  local gpuIngrs = state.gpuIngrs
  local gpuMatches = state.gpuMatches

  local gpuClsRecipe = state.gpuClsRecipe
  local gpuClsImg = state.gpuClsImg

  local inp
  local outp
  inp = {gpuImgs, gpuInstrs, gpuIngrs}
  if opts.semantic then
    outp = {gpuMatches,gpuClsImg,gpuClsRecipe}
  else
    outp = gpuMatches
  end

  local crit = state.crit

  -- patience > iter_swap
  if opts.patience ~=-1 and opts.iter_swap ~=1 then
    opts.iter_swap = -1
  end
  state.trainLoss = 0

  -- auxiliar losses for display only
  if opts.semantic then
    state.cosloss = 0
    state.classloss_img = 0
    state.classloss_rec = 0
  end

  local optimizer = optim.adam
  local optimState

  optimState = {
    learningRate = tonumber(opts.lr),
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8
  }

  local params, gradParams = model:getParameters()
  local function f() return err, gradParams end

  local function lr_setup(model,freeze_branch,d_rate)
    -- sets individual optimstate for each chunk of the network.
    layerParameters, layerGradParameters = model:parameters()
    layerOptimState = {}
    print (#layerParameters)
    sep = #layerParameters-opts.n_layer_trijoint*2 -- layer id from which we set base lr

    for i=1,#layerParameters do
      if freeze_branch == 'vision' then
        if i < sep then
          lr = 0
        else
          lr = tonumber(opts.lr)/d_rate
        end
      else
        if i < sep then
          lr = tonumber(opts.lr)/d_rate
        else
          lr = 0
        end
      end

      table.insert(layerOptimState, {
                                    learningRate = lr,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 1e-8
                                  }
                                )

    end
    return layerParameters,layerGradParameters,layerOptimState
  end

  if opts.patience ~= -1 or opts.iter_swap ~=-1 then
    frozen_branch = opts.freeze_first
    layerParameters,layerGradParameters,layerOptimState = lr_setup(model,frozen_branch,state.dswap)
    state.dswap = opts.dec_lr
  end
  local function doTrain(batchInps, batchOuts)

    cutorch.synchronize()
    collectgarbage()

    state.t = state.t + 1

    batchImgs = batchInps[1]
    batchInstrs = batchInps[2]
    batchIngrs = batchInps[3]

    gpuImgs:resize(batchImgs:size()):copy(batchImgs)
    gpuInstrs:resize(batchInstrs:size()):copy(batchInstrs)
    gpuIngrs:resize(batchIngrs:size()):copy(batchIngrs)

    if opts.semantic then
      batchMatches = batchOuts[1]
      batchClsImg = batchOuts[2]
      batchClsRecipe = batchOuts[3]

      gpuMatches:resize(batchMatches:size()):copy(batchMatches)
      gpuClsRecipe:resize(batchClsRecipe:size()):copy(batchClsRecipe)
      gpuClsImg:resize(batchClsImg:size()):copy(batchClsImg)
    else
      gpuMatches:resize(batchOuts:size()):copy(batchOuts)
    end

    model:forward(inp)
    model:zeroGradParameters()
    if opts.semantic then
      n1 = 1
      n2 = 2
      n3 = 3
      state.cosloss = state.cosloss + crit[1]:forward(model.output[n1], outp[n1])
      state.classloss_img = state.classloss_img + crit[2]:forward(model.output[n2], outp[n2])
      state.classloss_rec = state.classloss_rec + crit[2]:forward(model.output[n3], outp[n3])
      err = crit[3]:forward(model.output, outp)
      state.trainLoss = state.trainLoss + err
      crit[3]:backward(model.output, outp)
      model:backward(inp, crit[3].gradInput)
    else
      err = crit:forward(model.output, outp)
      state.trainLoss = state.trainLoss + err
      crit:backward(model.output, outp)
      model:backward(inp, crit.gradInput)
    end

    if opts.patience ~= -1 or opts.iter_swap ~= -1 then
      -- updates for modular learning rates
      for j=1,#layerParameters do

        local feval2 = function(x)
          return err, layerGradParameters[j]
        end
        optimizer(feval2, layerParameters[j], layerOptimState[j])
      end
    else
      optimizer(f, params, optimState)
    end

    cutorch.synchronize()
    if state.t ==1 then
      timer = torch.Timer()
    end

    -- Swap training of vision or trijoint branches of the model

    if state.valtrack >= opts.patience and opts.patience ~= -1 or opts.iter_swap ~=-1 and state.t % opts.iter_swap == 0 then
      if frozen_branch == 'vision' then
        print ('Swap from vision to trijoint learning.','lr:',string.format(tonumber(opts.lr)/state.dswap))
        frozen_branch = 'trijoint'
      else
        frozen_branch = 'vision'
        print ('Swap from trijoint to vision learning.','lr:',string.format(tonumber(opts.lr)/state.dswap))
      end
      layerParameters,layerGradParameters,layerOptimState = lr_setup(model,frozen_branch,state.dswap)
      state.dswap = state.dswap*opts.dec_lr
      state.valtrack = 0
    end

    if state.t % opts.dispfreq == 0 then
      print(state.t,'Time:' .. timer:time().real .. ' seconds.')

      if opts.semantic then
        print(string.format('Cos loss: %g', state.cosloss /opts.dispfreq),string.format('Class loss (im): %g', state.classloss_img /opts.dispfreq), string.format('Class loss (rec): %g', state.classloss_rec /opts.dispfreq),string.format('Total loss: %g', state.trainLoss /opts.dispfreq))

        state.cosloss = 0
        state.classloss_img = 0
        state.classloss_rec = 0

      else
        print(string.format('Train loss: %g', state.trainLoss /opts.dispfreq))
      end

      state.trainLoss = 0
      timer = torch.Timer()

    end
  end

  return function()
    model:training()
    workers:addjob(function() return dataLoader:makebatch() end, doTrain)
  end
end
