require 'cunn'
local ffi=require 'ffi'

function makeDataParallel(model, opts)
   if opts.ngpus > 1 then
      print('converting module to nn.DataParallelTable')
      assert(opts.ngpus <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, opts.ngpus do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end

   end
   cutorch.setDevice(opts.gpu)
   return model
end

local function cleanDPT(module,opts)
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opts.gpu)
   newDPT:add(module:get(1), opts.gpu)
   return newDPT
end

function saveParallel(filename,model,opts)
  -- model is: sequential(paralleltable(dptable,seq,seq),concattable,concattable)
  temp_model = nn.Sequential()
  for i,m in ipairs(model.modules) do
    if torch.type(m) == 'nn.ParallelTable' then
      branch = nn.ParallelTable()
      for ii,mm in ipairs(m.modules) do
        if torch.type(mm) == 'nn.DataParallelTable' then
          branch:add(cleanDPT(mm,opts))
        else
          branch:add(mm)
        end
      end
      temp_model:add(branch)
    else
      temp_model:add(m)
    end
  end
  torch.save(filename, temp_model:clearState())
end

function loadParallel(model,opts)

  if opts.backend == 'cudnn' then
    require 'cudnn'
  end
  -- model is: sequential(paralleltable(dptable,seq,seq),concattable,concattable)
  for i,m in ipairs(model.modules) do
    if torch.type(m) == 'nn.ParallelTable' then
      for ii,mm in ipairs(m.modules) do
        if torch.type(mm) == 'nn.DataParallelTable' then
          model.modules[i].modules[ii] = makeDataParallel(mm:get(1):float(), opts)
        end
      end
    end
  end
  return model
end
