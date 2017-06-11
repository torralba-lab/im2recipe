local args = {}

DATA_DIR = paths.thisfile('data')
print(DATA_DIR)

function args.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')

  cmd:option('-test', 0, 'test mode (0 or 1)')
  cmd:option('-extract', 'test', 'extract features on indicated set')
  cmd:option('-backend', 'cudnn', '(nn|cunn|cudnn)')
  cmd:option('-seed', 4242, 'Manual random seed')

  -- Data
  cmd:option('-dataset', DATA_DIR..'/data.h5', '')
  cmd:option('-ingrW2V', DATA_DIR..'/text/vocab.bin')

  -- training params
  cmd:option('-mismatchFreq', 0.8, '')
  cmd:option('-nworkers', 4, 'number of data loading threads')
  cmd:option('-gpu', 1, 'gpu id')
  cmd:option('-ngpus', 4, 'multigpu')
  cmd:option('-batchSize', 150, 'mini-batch size')

  -- Vision model
  cmd:option('-imsize',224, 'size of image crop(square)')
  cmd:option('-imstore',256, 'size of images saved in disk')
  cmd:option('-net', 'resnet', 'resnet or vgg')

  cmd:option('-patience', 3, 'Number of validation steps to wait until swap, -1 means no swap - all trained at once')
  cmd:option('-iter_swap', -1, 'Fix number of iterations between freeze switches. -1 means no swap.')
  cmd:option('-dec_lr', 1, 'Divide learning rate by value every time we swap (value of 1 will leave as is)')
  cmd:option('-n_layer_trijoint', 7, 'Number of layers in trijoint model (5/7 for semantic=0/semantic=1)')
  cmd:option('-freeze_first', 'vision', 'Branch to freeze first (trijoint|vision)')
  --vgg16 model
  cmd:option('-proto',DATA_DIR..'/vision/VGG_ILSVRC_16_layers_deploy.prototxt', 'deploy file')
  cmd:option('-caffemodel',DATA_DIR..'/vision/VGG_ILSVRC_16_layers.caffemodel', 'caffe model file')
  cmd:option('-remove', 2, 'number of layers to remove after loading vgg')
  --resnet model
  cmd:option('-resnet_model',DATA_DIR..'/vision/resnet-50.t7', 'resnet torch model file')

  -- Trijoint model
  cmd:option('-embDim', 1024, '')
  cmd:option('-nRNNs', 1, '')
  cmd:option('-srnnDim', 1024, '')
  cmd:option('-irnnDim', 300, '')
  cmd:option('-imfeatDim', 2048, '')
  cmd:option('-stDim', 1024, '')
  cmd:option('-ingrW2VDim', 300)
  cmd:option('-maxSeqlen', 20, '')
  cmd:option('-maxIngrs', 20, '')
  cmd:option('-maxImgs',5,'max number of images per sample')

  -- Semantic regularization
  cmd:option('-semantic', 1, 'Bool to include semantic branch')
  cmd:option('-numClasses', 1048, 'Number of classes')
  cmd:option('-cosw', 0.98, 'Weight to cosine criterion loss')
  cmd:option('-clsw', 0.01, 'NLL weight (x2)')

  -- Training
  cmd:option('-lr', 0.0001, 'base learning rate')
  cmd:option('-optim', 'adam', 'optimizer (adam|sgd)')
  cmd:option('-niters', -1, 'number of iterations for which to run (-1 is forever)')
  cmd:option('-dispfreq', 1000, 'number of iterations between printing train loss')
  cmd:option('-valfreq', 10000, 'number of iterations between validations. Snapshot will be saved for all validations(if increases performance).')

  -- Saving & loading
  --
  cmd:option('-snapfile', 'snaps/resnet_reg', 'snapshot file prefix')
  cmd:option('-loadsnap', '', 'file from which to load model')
  cmd:option('-rundesc', '', 'description of what is being tested')

  cmd:text()

  return cmd:parse(arg or {})
end

return args
