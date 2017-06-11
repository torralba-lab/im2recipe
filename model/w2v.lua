require 'paths'

local function readWord(fw2v)
  local chars = {}
  while true do
    local char = fw2v:readChar()
    if char == 0 or char == 32 then
      break
    elseif char ~= 10 then
      chars[#chars+1] = char
    end
  end

  return torch.CharStorage(chars):string()
end

return function(w2vBin)
  local cachePath = paths.thisfile(''..paths.basename(w2vBin, '.bin')..'.t7')
  if paths.filep(cachePath) then return table.unpack(torch.load(cachePath)) end

  local fw2v = torch.DiskFile(w2vBin)
  local nwords, embDim = fw2v:readInt(), fw2v:readInt()

  local i2w = {}
  local wvecs = torch.FloatTensor(nwords, embDim):zero()

  fw2v:binary()
  for i=1,nwords do
    i2w[i] = readWord(fw2v)
    wvecs[i] = torch.FloatTensor(fw2v:readFloat(embDim))
  end

  wvecs:cdiv(wvecs:norm(2, 2):expandAs(wvecs))

  torch.save(cachePath, {wvecs, i2w})

  return wvecs, i2w
end
