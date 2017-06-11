require 'paths'
require 'image'
require 'hdf5'

function ImageNormalize(img,meanstd)
  img = img:clone()
  img = img[{ 1, {}, {}, {} }]
  for i=1,3 do
     img[i]:add(-meanstd.mean[i])
     img[i]:div(meanstd.std[i])
  end
  img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image
  return img
end

-- reads an image disk
-- if it fails to read the image, it will use a blank image
-- and write to stdout about the failure
-- this means the program will not crash if there is an occassional bad apple
function loadImage(path,loadSize)
  local ok,input = pcall(image.load, path, 3, 'float')
  if not ok then
     print('warning: failed loading: ' .. path)
     input = torch.zeros(3, loadSize, loadSize)
  else
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
        input = image.scale(input, loadSize, loadSize * iH / iW)
    else
        input = image.scale(input, loadSize * iW / iH, loadSize)
    end
  end

  return input
end

function improc(imsize,input)
  input = input:view(input:size(2), input:size(3), input:size(4))
   -- random or center cropping & random flip image
   collectgarbage()
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = imsize
   local oH = imsize
   local h1
   local w1
   h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   w1 = math.ceil(torch.uniform(1e-2, iW-oW))

   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end

   return out
end

function groupByLen(data)
  local rlens = data.rlens

  local indsByLen = {}
  for i=1,rlens:size(1) do
    local rlen = rlens[i]

    local ibrl = indsByLen[rlen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[rlen] = ibrl
  end

  local lengths = {}
  for len,_ in pairs(indsByLen) do
    lengths[#lengths+1] = len
  end
  table.sort(lengths)

  local lenFreqs = torch.zeros(#lengths)
  for i=1,#lengths do
    lenFreqs[i] = #indsByLen[lengths[i]]
  end
  lenFreqs:div(lenFreqs:sum())

  data.lengths = lengths
  data.indsByLen = indsByLen
  data.lenFreqs = lenFreqs
end
