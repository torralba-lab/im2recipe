require 'nngraph'
require 'loadcaffe'
require 'nnlr'

-------------------------------------------------------------------------------
-- Create models

function get_vision_branch(opts)

  local cnn
  if opts.net == 'vgg' then
    cnn = get_vgg(opts)
    cnn = make_net(cnn,opts)
  else
    cnn = torch.load(opts.resnet_model)
    cnn:remove() -- remove last layer
  end
  return cnn
end

function get_vgg(opts)
  local cnn = loadcaffe.load(opts.proto, opts.caffemodel,opts.backend)
  -- remove layers after fc7
  for i=1,opts.remove do
    cnn:remove()
  end
  return cnn
end

function make_net(cnn,opts)

  -- copy over the first layer_num layers of the CNN
 local cnn_part = nn.Sequential()
 for i = 1, #cnn do
   local layer = cnn:get(i)

   if i == 1 then
     -- convert kernels in first conv layer into RGB format instead of BGR,
     -- which is the order in which it was trained in Caffe

     local w = layer.weight:clone()
     -- swap weights to R and B channels
     print('converting first layer conv filters from BGR to RGB...')
     layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
     layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
   end

   cnn_part:add(layer)
 end
 return cnn_part
end

function build_model_finetune(opts,model)

  -- Function to add vision branch to a model trained with fixed vision features

  local im = nn.Identity()()
  local instrs = nn.Identity()()
  local ingrs = nn.Identity()()

  print('Composing model with vision and loaded snapshot')

  local cnn = get_vision_branch(opts)
  cnn = makeDataParallel(cnn, opts) -- parallelize vision branch if needed
  local fc7 = cnn({im})

  -- freeze loaded model
  model.accGradParameters = function() end

  output = model({fc7,instrs,ingrs})
  return nn.gModule({im,instrs,ingrs}, {output})
end
