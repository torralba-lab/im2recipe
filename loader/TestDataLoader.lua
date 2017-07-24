require 'paths'
require 'hdf5'
local TestDataLoader = torch.class('TestDataLoader')

require 'loader.utils'
function TestDataLoader:__init(dataTest, opts)

  self.batchSize = opts.batchSize
  self.data = dataTest

  groupByLen(dataTest)

  self.imfeatDim = opts.imfeatDim
  self.stDim = opts.stDim
  self.maxIngrs = opts.maxIngrs
  self.net = opts.net
  self.meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
  }
  self.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}
  self.imsize = opts.imsize
  self.imstore = opts.imstore
  self.dataset = opts.dataset
  collectgarbage()
end

function TestDataLoader:makebatch()
  collectgarbage()
  local partData = self.data

  if partData.lenFreqs:sum() == 0 then
    return nil
  end

  local seqlenInd = torch.multinomial(partData.lenFreqs, 1)[1]
  local seqlen = partData.lengths[seqlenInd]
  local seqlenInds = partData.indsByLen[seqlen]
  local batchSize = math.min(#seqlenInds, self.batchSize)

  imagefile = hdf5.open(self.dataset, 'r'):read('/ims_'..partData.partition)
  local selIndInds = torch.randperm(#seqlenInds):sub(1, batchSize):sort()
  local remainingInds = {}
  local riPtr = 1
  -- for all idxs with the selected legth
  for i=1,#seqlenInds do -- remove selected indices (draw w/o replacement)

    if riPtr > selIndInds:size(1) or i ~= selIndInds[riPtr] then

      remainingInds[#remainingInds+1] = seqlenInds[i]
    else
      -- if riPtr is smaller than the batch size, increase it
      riPtr = riPtr + 1
    end
  end
  partData.indsByLen[seqlen] = #remainingInds > 0 and remainingInds or nil
  if #remainingInds == 0 then
    partData.lenFreqs[seqlenInd] = 0
  end

  local batchIds = torch.CharTensor(batchSize, 11):zero()
  local batchImgs = torch.Tensor(batchSize, 3,self.imsize,self.imsize)
  local batchInstrs = torch.Tensor(seqlen, batchSize, self.stDim)
  local batchIngrs = torch.LongTensor(batchSize, self.maxIngrs)

  for i=1,batchSize do
    local selInd = seqlenInds[selIndInds[i]]

    batchIds[i]:sub(1, 10):copy(partData.ids[selInd])

    local mean_pixel = self.vgg_mean:view(1,3,1,1)

    local numims = partData.numims[selInd]
    local imid = torch.randperm(numims)[1]
    local impos = partData.impos[selInd][imid]

    local img = imagefile:partial({impos,impos},{1,3},{1,self.imstore},{1,self.imstore}):float()
    img = improc(self.imsize,img)
    img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image

    if self.net == 'resnet' then
      img:div(256)
      img = ImageNormalize(img,self.meanstd)
    else
      img:add(-1, mean_pixel:expandAs(img))
    end
    batchImgs[i] = img

    batchIngrs[i] = partData.ingrs[selInd]

    local rbp = partData.rbps[selInd]
    local rlen = partData.rlens[selInd]
    batchInstrs[{{1, seqlen}, i}] = partData.stvecs:narrow(1, rbp, rlen)
  end
  imagefile:close()
  return {batchImgs, batchInstrs, batchIngrs}, batchIds
end

function TestDataLoader:partitionSize(partition)
  return self.data.rlens:size(1)
end

function TestDataLoader:terminate() end
