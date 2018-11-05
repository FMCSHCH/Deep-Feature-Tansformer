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
require 'stn'
cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'results/model.net', 'subdirectory to save/log experiments in')
cmd:option('-load', '', 'load saved model as starting point')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | NAG')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 20, 'mini-batch size (1 = pure stochastic)')
cmd:option('-epochs', 50, 'number of epochs')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)') 
cmd:option('-type', 'cuda', 'use cuda')
cmd:option('-transfer', 'relu', 'activation function, options are: elu, relu')
cmd:option('-train', true, 'train the model')
cmd:option('-test', true, 'test the model')
cmd:option('-locnet', "30,60,30", 'Localization network')
cmd:option('-rot', false, 'rotation')
cmd:option('-sca', false, 'scale')
cmd:option('-tra', "", 'translation')
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

function networks.new_conv2(nbr_input_channels,nbr_output_channels,
                           multiscale, no_cnorm, filter_size)
  multiscale = multiscale or false
  no_cnorm = no_cnorm or false
  filter_size = filter_size or 3
  local padding_size = 1
  local pooling_size = 1
  local normkernel = image.gaussian1D(7)

  local conv

  local first = nn.Sequential()
  first:add(networks.modules.convolutionModule(nbr_input_channels,
                                      nbr_output_channels,
                                      filter_size, filter_size,
                                      1,1,
                                      padding_size, padding_size))
  first:add(networks.modules.nonLinearityModule())
  --first:add(networks.modules.poolingModule(pooling_size, pooling_size,
    --                                       pooling_size, pooling_size))
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
-- Gives the number of output elements for a table of convolution layers
-- Also returns the new height (=width) of the image
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

-- Creates a fully connection layer with the specified size.
function networks.new_fc(nbr_input, nbr_output)
  local fc = nn.Sequential()
  fc:add(nn.View(nbr_input))
  fc:add(nn.Linear(nbr_input, nbr_output))
  fc:add(networks.modules.nonLinearityModule())
  return fc
end

-- Creates a classifier with the specified size.
function networks.new_classifier(nbr_input, nbr_output)
  local classifier = nn.Sequential()
  classifier:add(nn.View(nbr_input))
  classifier:add(nn.Linear(nbr_input, nbr_output))
  return classifier
end


function networks.new_spatial_tranformer_new(locnet, rot, sca, tra,
                                         input_size, input_channels,
                                         no_cuda)
	
  input_size = input_size or networks.base_input_size
  input_channels = 512
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
  print ('nbr_params==')
print(nbr_params)
print ('init_bias==')
print (init_bias)
  local st = nn.Sequential()

   -- Create a localization network same as cnn but with downsampled inputs
  local localization_network = nn.Sequential()
   local branch3=nn.Sequential()
   local branch4=nn.Identity()
   local CVPR_network1=nn.ConcatTable()
   local CVPR_network=nn.Sequential()
  local conv1 = networks.new_conv2(128, 64,
                                           	 false, false, 3)
	local conv2 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 

	local conv3 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv4 = networks.new_conv2(64, 64,
                                           	 false, false, 3)										 
	local conv5 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv6 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv7 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv8 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv9 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv10 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
									 
											 
	local conv11 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv12 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv13 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv14 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv15 = networks.new_conv2(64, 128,
                                           	 false, false, 3)
		--[[											 
	local conv16 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv17 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv18 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv19 = networks.new_conv2(64, 64,
                                           	 false, false, 3)
	local conv20 = networks.new_conv2(64, 512,
                                           	 false, false, 3)	
    ]]											 
											 
  

    localization_network:add(networks.modules.poolingModule(2,2,2,2))
	branch3:add(conv1)
    branch3:add(conv2)
    branch3:add(conv3)
    branch3:add(conv4)
	branch3:add(conv5)
    branch3:add(conv6)
    branch3:add(conv7)
	branch3:add(conv8)
	branch3:add(conv9)
	branch3:add(conv10)
	
	branch3:add(conv11)
	branch3:add(conv12)
	branch3:add(conv13)
	branch3:add(conv14)
	branch3:add(conv15)
	--[[
	branch3:add(conv16)
	branch3:add(conv17)
	branch3:add(conv18)
	branch3:add(conv19)
	branch3:add(conv20)
	]]
	CVPR_network1:add(branch3)
	CVPR_network1:add(branch4)
	CVPR_network:add(CVPR_network1)
	CVPR_network:add(nn.CAddTable())
	
	
	-- new_network
	     local new_network_total=nn.Sequential()
		local new_network=nn.ConcatTable()
		local branch_1_new_network=nn.Sequential()
	    VGG_model = torch.load('VGG_inside.t7')
		for _ = 1,27 do
			VGG_model:remove()
			end
	    branch_1_new_network:add(VGG_model)
		branch_1_new_network:add(CVPR_network) 
		conv_128_512=networks.new_conv2(128, 512,
                                           	 false, false, 3)	
	    branch_1_new_network:add(conv_128_512)
	
		local branch_2_new_network=nn.Identity()
		new_network:add(branch_1_new_network)
		new_network:add(branch_2_new_network)
		new_network_total:add(new_network)
			
		new_network_total:add(nn.CAddTable())
	
	   localization_network:add(new_network_total)
	   
	   
	   local conv_output_size = networks.convs_noutput({conv_128_512}, input_size/2)
     -- local conv_output_size = networks.convs_noutput({conv3}, input_size/2)
      local fc = networks.new_fc(conv_output_size, nbr_elements[3])
     local classifier = networks.new_classifier(nbr_elements[3], nbr_params)
  -- Initialize the localization network (see paper, A.3 section)
  classifier:get(2).weight:zero()
  classifier:get(2).bias = torch.Tensor(init_bias)
  
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

  branch2:add(nn.AffineGridGeneratorBHWD(input_size,input_size))
  if not no_cuda then -- see (1) below
    branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
  end
  ct:add(branch1)
  ct:add(branch2)

  st:add(ct)
  local sampler = nn.BilinearSamplerBHWD()
  -- (1)
  -- The sampler lead to non-reproducible results on GPU
  -- We want to always keep it on CPU
  -- This does no lead to slowdown of the training
  if not no_cuda then
    sampler:type('torch.FloatTensor')
    -- make sure it will not go back to the GPU when we call
    -- ":cuda()" on the network later
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

--[[MODEL]]--
 model = nn.Sequential()
Trained_model = torch.load('Trained_Network_4.bin')
--L1=Trained_model:get(1)
L2=Trained_model:get(2)
--L3=Trained_model:get(3)
Trained_model:insert(L2,4)
--Trained_model:insert(L3,3)
model:add(Trained_model)
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
trainSetPath = '/data/hossamkasem/5-SVHN/5-Ours_multi_original/New_train_daset.bin'
trainSet = torch.checkpoint(trainSetPath) 
trainSize = trainSet.label:size()[1]
print(trainSet)
testSetPath = '/data/hossamkasem/5-SVHN/5-Ours_multi_original/test_dataset.bin'
testSet = torch.checkpoint(testSetPath) 
testSize = testSet.label:size()[1]
print(testSet)
-- subtract the mean of each image



-- set cuda
if opt.type == 'cuda' then
    model:cuda()
  criterion:cuda()
end

-- initialize params
parameters,gradParameters = model:getParameters()
--parameters:uniform(-0.1,0.1)	

 

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
  torch.save('/data/hossamkasem/5-SVHN/5-Ours_multi_original/results/train_confusion_matrix/Train_confusion_'..epochs..'.bin', accuracy)
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

  torch.save('/data/hossamkasem/5-SVHN/5-Ours_multi_original/results/test_confusion_matrix/Test_confusion_'..epochs..'.bin', accuracy)
   
end

for e = 1,opt.epochs do
     epochs=e
  if opt.train then train(epochs) end
  
  -- save/log current net
  if opt.save ~= '' then
  local filename = opt.save--paths.concat(opt.save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save('/data/hossamkasem/5-SVHN/5-Ours_multi_original/results/Trained_Network_'..epochs..'.bin', model)
  end

  if opt.test then test(epochs) end
end