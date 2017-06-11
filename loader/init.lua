require 'hdf5'
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

loader = {}

local function loadPartition(dsH5,partition,opts)
  local data = {}
  data.ids = dsH5:read('/ids_'..partition):all()
  data.stvecs = dsH5:read('/stvecs_'..partition):all()
  data.ingrs = dsH5:read('/ingrs_'..partition):all()
  data.rlens = dsH5:read('/rlens_'..partition):all()
  data.rbps = dsH5:read('/rbps_'..partition):all()
  data.numims = dsH5:read('/numims_'..partition):all()
  data.impos = dsH5:read('/impos_'..partition):all()
  if opts.semantic then
    data.classes = dsH5:read('/classes_'..partition):all()
  end
  data.partition = partition
  return data
end

function loader.init(opts)

  local nworkers = opts.nworkers
  local seed = opts.seed

  print('loading data')
  -- load these here because hdf5 isn't multi-threaded and also shared serialization
  local dataTrain, dataVal, dataExtract
  local dsH5 = hdf5.open(opts.dataset, 'r')

  if opts.test then
    dataExtract = loadPartition(dsH5,opts.extract,opts)
  else
    dataTrain = loadPartition(dsH5,'train',opts)
    dataVal = loadPartition(dsH5,'val',opts)
  end
  dsH5:close()
  --local soreor = torch.load(opts.soreor)
  print('data loaded')

  if nworkers > 0 then
    workers = threads.Threads(
      nworkers,
      function()
        require 'torch'
        if opts.test then
          require 'loader.TestDataLoader'
        else
          require 'loader.DataLoader'
        end
      end,
      function(tid)
        math.randomseed(seed + tid)
        torch.manualSeed(seed + tid)

        if opts.test then
          require 'loader.TestDataLoader'
          dataLoader = TestDataLoader(dataExtract, opts)
        else
          require 'loader.DataLoader'
          dataLoader = DataLoader(dataTrain, dataVal, opts)
        end
      end
    )
  else -- single threaded data loading. useful for debugging
    if opts.test then
      require 'loader.TestDataLoader'
      dataLoader = TestDataLoader(dataExtract, opts)
    else
      require 'loader.DataLoader'
      dataLoader = DataLoader(dataTrain, dataVal, opts)
    end
    workers = {}
    function workers:addjob(f1, f2) f2(f1()) end
    function workers:synchronize() end
    function workers:terminate() end
  end

  return workers
end

return loader
