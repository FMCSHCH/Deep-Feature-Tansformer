require 'nn'
require 'optim'
require 'torch'
require 'xlua'
require 'dp'
--require 'rnn'
require 'cutorch'
require 'cunn'
require 'image'
--require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'results/model.net', 'subdirectory to save/log experiments in')
cmd:option('-load', '/data/hossamkasem/5-SVHN/3-Single_ST_CNN/model.net', 'load saved model as starting point')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | NAG')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 50, 'mini-batch size (1 = pure stochastic)')
cmd:option('-epochs', 50, 'number of epochs')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)') 
cmd:option('-type', 'cuda', 'use cuda')
cmd:option('-transfer', 'relu', 'activation function, options are: elu, relu')
cmd:option('-train', true, 'train the model')
cmd:option('-test', true, 'test the model')
cmd:option('-locnet', "30,60,30", 'Localization network')
cmd:option('-rot',true , 'rotation')
cmd:option('-sca',true , 'scale')
cmd:option('-tra', true, 'translation')
cmd:option('-no_cuda', false, 'translation')


cmd:text()
opt = cmd:parse(arg or {})
print(opt)
local lengthClasses = 7
local maxDigits = lengthClasses - 2 -- class 7 is "> maxDigits"
local digitClasses = 10

local networks = {}
networks.modules = {}
networks.modules.convolutionModule = nn.SpatialConvolutionMM
networks.modules.poolingModule = nn.SpatialMaxPooling
networks.modules.nonLinearityModule = nn.ReLU
networks.base_input_size = 54
networks.nbr_classes = 10

local function convLayer(nInput, nOutput, stride)
  local kW = 5
  local kH = 5
  local padW = (kW - 1)/2
  local padH = (kH - 1)/2
  local layer = nn.Sequential()
  layer:add(nn.SpatialConvolution(nInput, nOutput, kW, kH, 1, 1, padW, padH))
  layer:add(nn.SpatialMaxPooling(2, 2, stride, stride, 1, 1))
  layer:add(nn.SpatialBatchNormalization(nOutput))
  layer:add(nn.ReLU())
  layer:add(nn.Dropout(0.2))
  if opt.type == 'cuda' then
    return layer:cuda()
  else
    return layer
  end
end

function networks.new_conv(nbr_input_channels,nbr_output_channels,
                           multiscale, no_cnorm, filter_size)
  multiscale = multiscale or false
  no_cnorm = no_cnorm or false
  filter_size = filter_size or 5
  local padding_size = 2
  local pooling_size = 2
  local normkernel = image.gaussian1D(7)

  local conv

  local first = nn.Sequential()
  first:add(networks.modules.convolutionModule(nbr_input_channels,
                                      nbr_output_channels,
                                      filter_size, filter_size,
                                      1,1,
                                      padding_size, padding_size))
  first:add(networks.modules.nonLinearityModule())
  first:add(networks.modules.poolingModule(pooling_size, pooling_size,
                                           pooling_size, pooling_size))
  if not no_cnorm then
    first:add(nn.SpatialContrastiveNormalization(nbr_output_channels,
                                                 norm_kernel))
  end

  if multiscale then
    conv = nn.Sequential()
    local second = networks.modules.poolingModule(pooling_size, pooling_size,
                                              pooling_size, pooling_size)

    local parallel = nn.ConcatTable()
    parallel:add(first)
    parallel:add(second)
    conv:add(parallel)
    conv:add(nn.JoinTable(1,3))
  else
    conv = first
  end

  return conv
end


function networks.convs_noutput(convs, input_size)
  input_size = input_size or networks.base_input_size
  -- Get the number of channels for conv that are multiscale or not
  local nbr_input_channels = convs[1]:get(1).nInputPlane or
                             convs[1]:get(1):get(1).nInputPlane
  local output = torch.Tensor(1,nbr_input_channels, input_size, input_size)
  for _, conv in ipairs(convs) do
    output = conv:forward(output)
  end
  return output:nElement(), output:size(3)
end


function networks.new_fc(nbr_input, nbr_output)
  local fc = nn.Sequential()
  fc:add(nn.View(nbr_input))
  fc:add(nn.Linear(nbr_input, nbr_output))
  fc:add(networks.modules.nonLinearityModule())
  return fc
end


function networks.new_classifier(nbr_input, nbr_output)
  local classifier = nn.Sequential()
  classifier:add(nn.View(nbr_input))
  classifier:add(nn.Linear(nbr_input, nbr_output))
  return classifier
end


function networks.new_spatial_tranformer(locnet, rot, sca, tra,
                                         input_size, input_channels,
                                         no_cuda)
   								 
   input_size = input_size or networks.base_input_size
  input_channels = input_channels or 1
  require 'stn'
  local nbr_elements = {}
  for c in string.gmatch(locnet, "%d+") do
    nbr_elements[#nbr_elements + 1] = tonumber(c)
  end


  -- Get number of params and initial state
  local init_bias = {}
  local nbr_params = 0
  if rot then
    nbr_params = nbr_params + 1
    init_bias[nbr_params] = 0
  end
  if sca then
    nbr_params = nbr_params + 1
    init_bias[nbr_params] = 1
  end
  if tra then
    nbr_params = nbr_params + 2
    init_bias[nbr_params-1] = 0
    init_bias[nbr_params] = 0
  end
  if nbr_params == 0 then
    -- fully parametrized case
    nbr_params = 6
    init_bias = {1,0,0,0,1,0}
  end

  local st = nn.Sequential()

  -- Create a localization network same as cnn but with downsampled inputs
  local localization_network = nn.Sequential()
  local conv1 = networks.new_conv(input_channels, nbr_elements[1], false, true)
  local conv2 = networks.new_conv(nbr_elements[1], nbr_elements[2], false, true)
  local conv_output_size = networks.convs_noutput({conv1, conv2}, input_size/2)
  local fc = networks.new_fc(conv_output_size, nbr_elements[3])
  local classifier = networks.new_classifier(nbr_elements[3], nbr_params)
  -- Initialize the localization network (see paper, A.3 section)
  classifier:get(2).weight:zero()
  classifier:get(2).bias = torch.Tensor(init_bias)

  localization_network:add(networks.modules.poolingModule(2,2,2,2))
  localization_network:add(conv1)
  localization_network:add(conv2)
  localization_network:add(fc)
  localization_network:add(classifier)

  -- Create the actual module structure
  local ct = nn.ConcatTable()
  local branch1 = nn.Sequential()
  branch1:add(nn.Transpose({3,4},{2,4}))
  if not no_cuda then -- see (1) below
    branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  end
  local branch2 = nn.Sequential()
  branch2:add(localization_network)
  branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))
  branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
  if not no_cuda then -- see (1) below
    branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  end
  ct:add(branch1)
  ct:add(branch2)

  st:add(ct)
  local sampler = nn.BilinearSamplerBHWD()
    if not no_cuda then
    sampler:type('torch.FloatTensor')
       sampler.type = function(type)
      return self
    end
    st:add(sampler)
    st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
  else
    st:add(sampler)
  end
  st:add(nn.Transpose({2,4},{3,4}))

  return st
end



--[[EXTRACTOR - extracts a feature vector H ]]--
extractor = nn.Sequential()
extractor:add(convLayer(1, 48, 2))
extractor:add(convLayer(48, 64, 1))
extractor:add(convLayer(64, 128, 2))
extractor:add(convLayer(128, 160, 1))
extractor:add(convLayer(160, 192, 2))
extractor:add(convLayer(192, 192, 1))
extractor:add(convLayer(192, 192, 2))
extractor:add(convLayer(192, 192, 1))
extractor:add(nn.View(192*7*7):setNumInputDims(3))
extractor:add(nn.Linear(192*7*7, 3072))
if opt.transfer == 'elu' then
  extractor:add(nn.ELU())
elseif opt.transfer == 'relu' then
  extractor:add(nn.ReLU())
end
extractor:add(nn.Linear(3072, 4096)) -- H

--[[CLASSIFIER - classifies one digit ]]--
local function classifier(classes)
  local classifier = nn.Sequential()
  classifier:add(nn.Linear(4096, classes))
  classifier:add(nn.LogSoftMax())
  if opt.type == 'cuda' then
    return classifier:cuda()
  else
    return classifier
  end
end

--[[SEQUENCER - classifies the length and all digits ]]--
sequencer = nn.ConcatTable()
sequencer:add(lengthPredictor)
sequencer:add(classifier(lengthClasses)) -- length predictor
for i = 1, maxDigits do
  sequencer:add(classifier(digitClasses)) -- digit class predictor
end

if opt.load ~= '' then
  Trained_model = torch.load(opt.load)
end
--[[MODEL]]--
model = nn.Sequential()
model:add(networks.new_spatial_tranformer(opt.locnet,
                                                  opt.rot, opt.sca, opt.tra,
                                                  nil, nil,
                                                  opt.no_cuda))
model:add(Trained_model)
--model:add(extractor) -- H
--model:add(sequencer)
print(model)

   


--[[LOSS]]--
criterion = nn.ClassNLLCriterion()
			

--[[ OPTIMIZATION ]]--
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-7,
  nesterov = false,
  dampening = 0
}
optimMethod = optim.sgd


-- loading datasets
trainSetPath = '/data/hossamkasem/5-SVHN/3-Single_ST_CNN/train_dataset.bin'
trainSet = torch.checkpoint(trainSetPath) 
trainSize = trainSet.label:size()[1]
print(trainSet)
testSetPath = '/data/hossamkasem/5-SVHN/3-Single_ST_CNN/test_dataset.bin'
testSet = torch.checkpoint(testSetPath) 
testSize = testSet.label:size()[1]
print(testSet)
-- subtract the mean of each image

for i = 1, trainSize do
  local mean = trainSet.data[i]:mean()
  trainSet.data[i]:add(-mean)
end
for i = 1, testSize do
  local mean = testSet.data[i]:mean()
  testSet.data[i]:add(-mean)
end

-- set cuda
if opt.type == 'cuda' then
  extractor:cuda()
  sequencer:cuda()
  model:cuda()
  criterion:cuda()
end

-- initialize params
parameters,gradParameters = model:getParameters()
parameters:uniform(-0.1,0.1)	



TrainError = 0

local function train(epochs)
  -- epochs tracking
  epochs = epochs
  


  model:training() -- for dropout
  model:zeroGradParameters()
	
  local confusion = optim.ConfusionMatrix(10)
  
  shuffle = torch.randperm(trainSize)
  for t = 1, trainSize - opt.batchSize, opt.batchSize do
    -- display progress
    xlua.progress(t, trainSize)
		
    -- get batch
    local inputs = trainSet.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
    local targets = trainSet.label:index(1, shuffle:sub(t, t + opt.batchSize - 1):long()):transpose(1,2)  
		
    -- get sequences lengths (actually length+1)
    local _, length = torch.max(targets, 1)
    length = length[1]		
    length[length:gt(maxDigits)] = maxDigits
    
    --image.save('/home/itaic/Documents/test.png', inputs[1])
    --print(targets[{{},{1}}])
    
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end
    
    -- evaluation function for optim
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      
      -- reset gradients
      gradParameters:zero()
      
      -- batch error accumulator
      local f = 0
			
      -- forward
      local output = model:forward(inputs)
			
      -- get error from criterion for length net
      local gradInput = {}
      if opt.type == 'cuda' then
        gradInput[1] = torch.Tensor(opt.batchSize, lengthClasses):zero():cuda()
        for i = 1,maxDigits do
          gradInput[i+1] = torch.Tensor(opt.batchSize, 10):zero():cuda()
        end
      else
        gradInput[1] = torch.Tensor(opt.batchSize, lengthClasses):zero()
        for i = 1,maxDigits do
          gradInput[i+1] = torch.Tensor(opt.batchSize, 10):zero()
        end
      end
      
      --print(targets)
      for b = 1, opt.batchSize do
        -- get gradients for the length tower
        local err = criterion:forward(output[1][b], length[b])
        gradInput[1][b] = criterion:backward(output[1][b], length[b])    
        f = f + err
        -- get gradients for each one of the digit towers
        for i = 1, length[b]-1 do
          local err = criterion:forward(output[i+1][b], targets[i][b])
          confusion:add(output[i+1][b], targets[i][b])
          gradInput[i+1][b] = criterion:backward(output[i+1][b], targets[i][b])
          f = f + err
        end
      end
      model:backward(inputs, gradInput)
      
      -- average gradients and error
      gradParameters:div(opt.batchSize)
      f = f / opt.batchSize
			
      TrainError = f
      
      
      --print(confusion)
      --print('Train Error = ' .. TrainError .. '\n')  
      return f, gradParameters
    end
		
    optimMethod(feval, parameters, optimState)
    
    collectgarbage('collect')

  end
  print(confusion)
  accuracy=confusion.totalValid * 100
  torch.save('/data/hossamkasem/5-SVHN/3-Single_ST_CNN/results/train_confusion_matrix/Train_confusion_'..epochs..'.bin', accuracy)
  print('Train Error = ' .. TrainError .. '\n')  
end

local function test(epochs)
  epochs = epochs

  model:evaluate()
  
  local confusion = optim.ConfusionMatrix(10)
  
  shuffle = torch.randperm(testSize)
  for t = 1, testSize - opt.batchSize, opt.batchSize do
    -- display progress
    xlua.progress(t, testSize)
		
    -- get batch
    local inputs = testSet.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
    local targets = testSet.label:index(1, shuffle:sub(t, t + opt.batchSize - 1):long()):transpose(1,2)  
		
    -- get sequences lengths (actually length+1)
    local _, length = torch.max(targets, 1)
    length = length[1]		
    length[length:gt(maxDigits)]=maxDigits
    --image.save('/home/itaic/Documents/test.png', inputs[1])
    --print(targets[{{},{1}}])
    
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end
    
    -- forward
    local output = model:forward(inputs)
			
    for b = 1, opt.batchSize do
      for i = 1, length[b]-1 do
        confusion:add(output[i+1][b], targets[i][b])
      end
    end  
  end
  print(confusion)
  accuracy=confusion.totalValid * 100

  torch.save('/data/hossamkasem/5-SVHN/3-Single_ST_CNN/results/test_confusion_matrix/Test_confusion_'..epochs..'.bin', accuracy)
   
end

for e = 1,opt.epochs do
     epochs=e
  if opt.train then train(epochs) end
  
  -- save/log current net
  if opt.save ~= '' then
  local filename = opt.save--paths.concat(opt.save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save('/data/hossamkasem/5-SVHN/3-Single_ST_CNN/results/Trained_Network_'..epochs..'.bin', model)
  end

  if opt.test then test(epochs) end
end