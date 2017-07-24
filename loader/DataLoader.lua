require 'paths'
require 'image'
require 'hdf5'

local DataLoader = torch.class('DataLoader')
require 'loader.utils'
function DataLoader:__init(dataTrain, dataVal, opts)

  self.batchSize = opts.batchSize
  self.mismatchFreq = opts.mismatchFreq -- percent of time a misaligned image should be presented
  self.dataTrain = dataTrain
  self.dataVal = dataVal
  --self.eor = soreor[2]:view(1, -1)

  groupByLen(dataTrain)
  groupByLen(dataVal)

  self.imfeatDim = opts.imfeatDim
  self.stDim = opts.stDim
  self.maxIngrs = opts.maxIngrs
  self.semantic = opts.semantic
  self.dataset = opts.dataset
  self.net = opts.net
  self.imstore = opts.imstore
  self.meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
  }
  self.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}
  self.imsize = opts.imsize
  torch.manualSeed(opts.seed)

  self.dsH5 = hdf5.open(self.dataset, 'r')

  collectgarbage()
end

function DataLoader:makebatch(partition)
  collectgarbage()
  partData = partition == 'val' and self.dataVal or self.dataTrain
  local seqlen = partData.lengths[torch.multinomial(partData.lenFreqs, 1)[1]]
  local seqlenInds = partData.indsByLen[seqlen]
  local batchSize = math.min(#seqlenInds, self.batchSize)

  local imagefile = self.dsH5:read('/ims_'..partData.partition)
  local selIndInds = torch.randperm(#seqlenInds) -- indices of indices in the batch
  local batchIds = torch.CharTensor(batchSize, 11)
  local batchImgs
  batchImgs = torch.Tensor(batchSize,3,self.imsize,self.imsize)

  local batchInstrs = torch.Tensor(seqlen, batchSize, self.stDim)
  local batchIngrs = torch.LongTensor(batchSize, self.maxIngrs)
  local batchMatches = torch.CharTensor(batchSize) -- whether image belongs to instrs
  local batchClsRecipe = torch.LongTensor(batchSize)
  local batchClsImg = torch.LongTensor(batchSize)

  --batchInstrs[-1]:copy(self.eor:expand(batchSize, self.stDim))

  local batchMaxIngrs = 0

  for i=1,batchSize do
    local selInd = seqlenInds[selIndInds[i]]

    batchIds[i]:sub(1, 10):copy(partData.ids[selInd])
    if self.semantic then
      batchClsRecipe[i] = partData.classes[selInd]
      batchClsImg[i] = partData.classes[selInd]
    end
    local match = math.random() > self.mismatchFreq
    batchMatches[i] = match and 1 or -1
    local position
    if match then
      position = selInd
    else
      local mismatch = selInd
      while mismatch == selInd do
        mismatch = torch.random(partData.ids:size(1))
      end
      position = mismatch
    end

    local mean_pixel = self.vgg_mean:view(1,3,1,1)

    -- READING FROM JPG FILE
    --name = torch.serialize(partData.images[mismatch]:clone(),'ascii')
    --local img = loadImage(name:sub(74,148),self.imstore)

    -- Select random image from list of images for that sample
    local numims = partData.numims[position]
    local imid = torch.randperm(numims)[1]
    local impos = partData.impos[position][imid]

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

    if self.semantic then
      batchClsImg[i] = partData.classes[position] -- replace class with mismatched class
    end

    local rbp = partData.rbps[selInd]
    local rlen = partData.rlens[selInd]

    local svecs = partData.stvecs:narrow(1, rbp, rlen)
    batchInstrs[{{1, seqlen}, i}] = svecs

    batchIngrs[i] = partData.ingrs[selInd]
  end
  if self.semantic then
    return {batchImgs, batchInstrs, batchIngrs}, {batchMatches,batchClsImg,batchClsRecipe}, batchIds
  else
    return {batchImgs, batchInstrs, batchIngrs}, batchMatches, batchIds
  end
end

function DataLoader:partitionSize(partition)
  local ds = partition == 'train' and self.dataTrain or self.dataVal
  return ds.rlens:size(1)
end

function DataLoader:terminate()
  self.dsH5:close()
end
