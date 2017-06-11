require 'cutorch'
require 'cunn'
require 'cudnn'
require 'sys'
require 'ffi'
require 'nn'

return function(model, workers, opts, state)
  local gpuImgs = state.gpuImgs
  local gpuInstrs = state.gpuInstrs
  local gpuIngrs = state.gpuIngrs
  local semantic = opts.semantic
  local idsTab = {}
  local imgEmbsTab = {}
  local instrEmbsTab = {}
  local clsImgTab = {}
  local clsRecTab = {}
  local batchSize = opts.batchSize
  local semantic = opts.semantic
  local extract = opts.extract
  local inp = {gpuImgs, gpuInstrs, gpuIngrs}


  return function()
    model:evaluate()

    local TESTBATCHES

    if batchSize == 512 then
      TESTBATCHES = 1000
    else
      TESTBATCHES = 3000
    end

    if extract=='train' then TESTBATCHES = TESTBATCHES*5 end
    print(TESTBATCHES)
    for i=1,TESTBATCHES do
      workers:addjob(
        function() return dataLoader:makebatch() end,
        function(batchInps, batchIds)
          if batchInps == nil then
            return
          end

          batchImgs = batchInps[1]
          batchInstrs = batchInps[2]
          batchIngrs = batchInps[3]

          gpuImgs:resize(batchImgs:size()):copy(batchImgs)
          gpuInstrs:resize(batchInstrs:size()):copy(batchInstrs)
          gpuIngrs:resize(batchIngrs:size()):copy(batchIngrs)

          local embs = model:forward(inp)

          local batchImgEmbs
          local batchInstrEmbs
          local batchClsImg
          local batchClsRec

          if semantic then

            n1 = 1
            n2 = 2
            n3 = 3

            batchImgEmbs = embs[n1][1]:double()
            batchInstrEmbs = embs[n1][2]:float()

            batchClsImg = embs[n2]:float()
            batchClsRec = embs[n3]:float()

            idsTab[#idsTab+1] = batchIds
            imgEmbsTab[#imgEmbsTab+1] = embs[n1][1]:float()
            instrEmbsTab[#instrEmbsTab+1] = embs[n1][2]:float()

            clsImgTab[#clsImgTab+1] = embs[n2]:float()
            clsRecTab[#clsRecTab+1] = embs[n3]:float()

          else
            batchImgEmbs = embs[1]:double()
            batchInstrEmbs = embs[2]:float()

            idsTab[#idsTab+1] = batchIds

            imgEmbsTab[#imgEmbsTab+1] = embs[1]:float()
            instrEmbsTab[#instrEmbsTab+1] = embs[2]:float()
          end
        end
      )
    end
    workers:synchronize()

    ids = torch.cat(idsTab, 1):char()
    torch.save('results/'..opts.extract ..'_ids.t7', ids)

    imgEmbs = torch.cat(imgEmbsTab, 1)
    instrEmbs = torch.cat(instrEmbsTab, 1)
    if opts.semantic then
      clsImg = torch.cat(clsImgTab, 1)
      clsRec = torch.cat(clsRecTab, 1)
      torch.save('results/'..opts.extract ..'_img_cls.t7', clsImg)
      torch.save('results/'..opts.extract ..'_rec_cls.t7', clsRec)
    end
    torch.save('results/'..opts.extract ..'_img_embs.t7', imgEmbs)
    torch.save('results/'..opts.extract ..'_instr_embs.t7', instrEmbs)

  end
end
