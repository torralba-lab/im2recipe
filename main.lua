require 'torch'
require 'cutorch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'drivers.CallbackQueue'

local args = require 'args'
local loader = require 'loader'
local drivers = require 'drivers'

local opts = args.parse(arg)

print(opts)
paths.dofile('drivers/utils.lua')
paths.dofile('model/trijoint.lua')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opts.seed)
math.randomseed(opts.seed)

opts.test = opts.test ~= 0
opts.semantic = opts.semantic ~= 0
opts.finetune = opts.finetune ~= 0

local model
if paths.filep(opts.loadsnap) then
  print('Loading model from '..opts.loadsnap)
  require 'dpnn'
  model = torch.load(opts.loadsnap) --load previously trained model
  model = loadParallel(model, opts)
else
  paths.dofile('model/trijoint.lua')
  model = get_trijoint(opts)
end

model:cuda()
model:training()
if opts.test then
  opts.nworkers = 1
end

local workers = loader.init(opts)

local train, val, snap, test = drivers.init(model, workers, opts)

if opts.test then
  print('Testing model')
  test()
  os.exit()
end

-- set up callbacks
local cbq = CallbackQueue(opts.startiter)

-- add end marker
cbq:add({cb=function() end, iter=opts.niters > 0 and opts.niters or math.huge})

-- add validation
cbq:add({cb=val, interval=opts.valfreq, iter=opts.valfreq, priority=math.huge})

-- add snapshotting
cbq:add({cb=snap, interval=opts.valfreq, iter=opts.valfreq})

collectgarbage()

-- val()
while #cbq > 0 do
  for t=1,cbq:waitTime() do train() end
  workers:synchronize()
  cbq:advance()
  for cb in cbq:pull() do cb() end
end

workers:addjob(function() dataLoader:terminate() end, function(n) nval = n end)
workers:terminate()
