require 'nn'
require 'rnn'
require 'cunn'
require 'cudnn'
require 'sys'

local loadW2V = paths.dofile(paths.thisfile('w2v.lua'))

local function mkIngrLut(opts)
  -- Ingredient lookup table
  local ingrW2V = loadW2V(opts.ingrW2V)
  local lut = nn.LookupTableMaskZero(ingrW2V:size(1)+1, ingrW2V:size(2))
  lut.weight:sub(3, -1):copy(ingrW2V) -- 1 is 0, 2 is </i>
  return lut
end

function get_trijoint(opts)

  local visionMLP
  local semantic_branch
  local visual_embedding

  -- Loads VGG or ResNet models
  paths.dofile('vision.lua')
  visionMLP = get_vision_branch(opts)

  -- Add embedding layer & activation
  visual_embedding = nn.Linear(opts.imfeatDim, opts.embDim)
  visionMLP:add(visual_embedding):add(cudnn.Tanh()):add(nn.Normalize(2))

  -- Parallelize vision branch into multiple gpus if needed
  paths.dofile('../drivers/utils.lua')
  visionMLP = makeDataParallel(visionMLP,opts)
  --end

  -- Skip-thoughts LSTM
  local stRNN = nn.Sequential()
    :add(cudnn.LSTM(opts.stDim, opts.srnnDim, opts.nRNNs))
    :add(nn.Select(1, -1))

  -- Ingredients Bidirectional LSTM
  local irnn = nn.SeqBRNN(opts.ingrW2VDim, opts.irnnDim, nil, nn.JoinTable(2, 2))
  irnn.maskzero = true
  local ingRNN = nn.Sequential()
    :add(mkIngrLut(opts))
    :add(nn.Transpose({1, 2}))
    :add(irnn)
    :add(nn.Select(1, -1))

  -- Build model with 3 inputs
  model = nn.Sequential()
    :add(nn.ParallelTable()
      :add(visionMLP)
      :add(stRNN)
      :add(ingRNN)
    )

    :add(nn.ConcatTable()
      :add(nn.SelectTable(1)) -- visual embedding
      :add(nn.Sequential() -- recipe embedding
        :add(nn.NarrowTable(2, 2)) -- select two last branches of parallel table
        :add(nn.JoinTable(2, 2)) -- concatenate two last branches in the second dim
        :add(nn.Linear(opts.irnnDim*2+opts.srnnDim, opts.embDim)) -- fc layer for recipe
        :add(cudnn.Tanh())
        :add(nn.Normalize(2))
      )
    )

  -- Adds classification layer
  if opts.semantic then
    local semantic_branch = nn.Linear(opts.embDim, opts.numClasses)
    local semantic_branch2 = semantic_branch:clone('weight', 'bias', 'gradWeight', 'gradBias')
    model:add(nn.ConcatTable() -- the embedding outputs
          :add(nn.ConcatTable()
              :add(nn.SelectTable(1)) -- image embedding
              :add(nn.SelectTable(2)) -- recipe embedding
            )
          :add(nn.Sequential()
            :add(nn.SelectTable(1)) -- image embedding
            :add(semantic_branch2)
            :add(nn.LogSoftMax())
          )
          :add(nn.Sequential()
            :add(nn.SelectTable(2)) -- recipe embedding
            :add(semantic_branch)
            :add(nn.LogSoftMax())
          )
        )
  end
  return model
end
