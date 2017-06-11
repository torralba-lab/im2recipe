require 'cutorch'
require 'cunn'
require 'cudnn'
require 'sys'

local VALBATCHES = 500

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

  return function()
    local valLoss = 0
    local correctSum = 0.0

    if opts.semantic then
      cosloss = 0
      classloss_img = 0
      classloss_rec = 0
    end

    model:evaluate()

    local saveseed = torch.random()
    math.randomseed(1234)
    torch.manualSeed(1234)

    for i=1,VALBATCHES do
      workers:addjob(
        function() return dataLoader:makebatch('val') end,
        function(batchInps, batchOuts)

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

          local preds = model:forward(inp)
          if opts.semantic then
            n1 = 1
            n2 = 2
            n3 = 3
            cosloss = cosloss + state.crit[1]:forward(model.output[n1], outp[n1])
            classloss_img = classloss_img + state.crit[2]:forward(model.output[n2], outp[n2])
            classloss_rec = classloss_rec + state.crit[2]:forward(model.output[n3], outp[n3])
            valLoss = valLoss + state.crit[3]:forward(preds, outp)
          else
            valLoss = valLoss + state.crit:forward(preds, outp)
          end


        end
      )
    end
    workers:synchronize()

    math.randomseed(saveseed)
    torch.manualSeed(saveseed)
    state.valPerf = -valLoss/VALBATCHES
    -- Save based on cosine loss only
    if opts.semantic then
      state.valPerf = -cosloss/VALBATCHES
      print(string.format('Cos loss: %g', cosloss/VALBATCHES),string.format('Class loss (im): %g', classloss_img/VALBATCHES),string.format('Class loss (rec): %g', classloss_rec/VALBATCHES),string.format('Total loss: %g', valLoss/VALBATCHES),'(val)')
    else
      print(string.format('Val loss: %g', valLoss/VALBATCHES))
    end

    collectgarbage()
  end
end
