# Copyright 2019 DIVERSIS Software. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

def get_scope_variable(scope, var, shape=None,initializer=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        v = tf.get_variable(var, initializer=initializer(shape),trainable=True)
    return v

def ConstructInferenceNetwork(InputSize,batchSize,layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,IMAGE_SIZE):
    # Generate Placeholders
    inputFramePlaceHolder = tf.placeholder(tf.float32, shape=[batchSize, InputSize[0], InputSize[1], InputSize[2]],name='inputFramePlaceHolder')
    labelPlaceHolder = tf.placeholder(tf.float32, shape=[batchSize, layerOutDimSize[-1]],name='labelPlaceHolder')
    dropoutInputPlaceHolder = tf.placeholder(tf.float32, shape=[],name='dropoutInputPlaceHolder')
    dropoutPoolPlaceHolder = tf.placeholder(tf.float32, shape=[len(layerOutDimSize)],name='dropoutPoolPlaceHolder')
    inputDistortionPlaceholder = tf.placeholder(tf.bool, shape=[],name='inputDistortionPlaceholder')
    # Construct distorted input if desired
    inputFramePlaceHolderResized = [tf.reshape(distorted_inputs(inputFramePlaceHolder[i,:,:,:],IMAGE_SIZE,inputDistortionPlaceholder),[1,IMAGE_SIZE,IMAGE_SIZE,InputSize[2]]) for i in range(inputFramePlaceHolder.shape[0])]
    inputFramePlaceHolderResized = tf.concat(inputFramePlaceHolderResized,axis=0)
    # Construct inference network structure    
    layerSize = len(layerOutDimSize)
    # Apply input dropout
    inputFrame = tf.nn.dropout(inputFramePlaceHolderResized,dropoutInputPlaceHolder,name="InputDropout")
    latentOut = tf.multiply(inputFrame,1.0)
    for lIndx in range(layerSize):
        print("**************** Layer-%d ****************"%lIndx)
        print("Input Tensor: ")
        print(latentOut)
        inputDim = latentOut.get_shape().as_list()
        outputDim = layerOutDimSize[lIndx]
        # if kernel is convolution
        if len(kernelSize[lIndx]) > 1:
            shapeW = [kernelSize[lIndx][0],kernelSize[lIndx][1],inputDim[-1],outputDim]
        else:
            # kernel is FC
            if len(inputDim) == 4:
                shapeW = [inputDim[1]*inputDim[2]*inputDim[3],outputDim]
            else:
                shapeW = [inputDim[-1],outputDim]
        weight = get_scope_variable('Layer-%d'%lIndx, 'Weight', shape=shapeW,initializer=tf.contrib.layers.xavier_initializer())
        bias = get_scope_variable('Layer-%d'%lIndx, 'Bias', shape=[outputDim],initializer=tf.zeros_initializer())
        print("Weight: ")
        print(weight)
        print("Bias: ")
        print(bias)
        # Construct layer
        lastLayer = (lIndx == (layerSize-1))
        latentOut = ConstructLayer(latentOut,weight,bias,strideSize[lIndx],'Layer-%d-OP'%lIndx,dropoutPoolPlaceHolder[lIndx],poolKernelSize[lIndx],poolSize[lIndx],lastLayer)
        print("Output Tensor: ")
        print(latentOut)
    print("******************************************")
        
    # Compute prediction metric
    softMaxOut = tf.nn.softmax(logits=latentOut,name="softMaxOut")
    correct_prediction = tf.equal(tf.argmax(softMaxOut,1,name="Argmax_softMaxOut"), tf.argmax(labelPlaceHolder,1,name="Argmax_Label"),name="CorrectPrediction")
    # Compute accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32,name="Cast_Accuracy"),name="Accuracy")
    
    placeHolders = {'inputFramePlaceHolder':inputFramePlaceHolder, 'labelPlaceHolder':labelPlaceHolder, 'dropoutInputPlaceHolder':dropoutInputPlaceHolder, 'dropoutPoolPlaceHolder':dropoutPoolPlaceHolder, 'inputDistortionPlaceholder':inputDistortionPlaceholder}
    
    return latentOut,accuracy,placeHolders

def ConstructLayer(layerInput,weight,bias,strideSize,nameScope,dropoutPoolPlaceHolder,poolKernelSize,poolSize,lastLayer):
    convSize = weight.get_shape().as_list()
    with tf.name_scope(nameScope):
        if len(convSize) == 4:
            outOp = tf.nn.conv2d(layerInput, weight, strides=[1,strideSize,strideSize,1], padding='SAME',name="ConvOP")
        else:
            layerInputFC = tf.reshape(layerInput,(layerInput.shape[0],-1))
            outOp = tf.matmul(layerInputFC, weight,name="MatMul")
        if poolSize > 1:
            outOp = tf.nn.max_pool(outOp, ksize=[1, poolKernelSize[0], poolKernelSize[1], 1], strides=[1, poolSize, poolSize, 1],padding='SAME', name='pool')

        layerOutput = tf.add(outOp,bias,name="BiasAdd")
        if lastLayer == False:
            layerOutput = tf.nn.dropout(layerOutput,dropoutPoolPlaceHolder)
            layerOutput = tf.nn.relu(layerOutput)

    return layerOutput

def ConstructOptimizer(output,labelPlaceHolder,momentum,weightDecay=None):
    learningRatePlaceHolder = tf.placeholder(tf.float32, shape=[],name='learningRatePlaceHolder')
    # Compute l2Loss
    if weightDecay is not None:
        l2LossList = [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
        l2Loss = tf.multiply(tf.add_n(l2LossList),weightDecay)
    else:
        l2Loss = tf.zeros(shape=[])
    # Compute coross entropy loss
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labelPlaceHolder, logits=output),name="CrossEntropy")
    # Compute totLoss used for training and accuracy metric        
    totLoss = tf.add_n([crossEntropy,l2Loss])
    # Generate optimizer operation
    with tf.variable_scope('Momentum-0', reuse=tf.AUTO_REUSE):       
        train_step = tf.train.MomentumOptimizer(learning_rate=learningRatePlaceHolder, momentum=momentum).minimize(totLoss)

    return train_step,totLoss,l2Loss,learningRatePlaceHolder

def GetTestAccuracy(sess,accuracyOp,data,labels,testBatchSize,frameBufferForTest,inputFramePlaceHolder,inputDistortionPlaceholder,labelPlaceHolder,dropoutInputPlaceHolder,dropoutPoolPlaceHolder):
    dataLen = data.shape[3]
    iternum = int(dataLen / testBatchSize)
    batchLabels = np.zeros((testBatchSize,labels.shape[0]))
    accuracy = 0
    for i in range(iternum):
        for j in range(testBatchSize):
            frameBufferForTest[j,:,:,:] = data[:,:,:,i*testBatchSize+j]
            batchLabels[j,:] = labels[:,i*testBatchSize+j]           
            
        # Determine feed_dict for testing accuracy
        feed_latent = {inputFramePlaceHolder:frameBufferForTest, inputDistortionPlaceholder:False, labelPlaceHolder:batchLabels, dropoutInputPlaceHolder:1.0, dropoutPoolPlaceHolder:np.ones((dropoutPoolPlaceHolder.shape[0]))}

        classiferAccuracyVal = sess.run(accuracyOp,feed_dict = feed_latent)
        accuracy += classiferAccuracyVal / iternum

    return accuracy

# We changed the files "distorted_inputs" in "https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py" 
# to generate "distorted_inputs" used in this file. 
def distorted_inputs(image,imSize,inputDistortionPlaceholder):
	with tf.name_scope('data_augmentation'):
		height = imSize
		width = imSize

		if inputDistortionPlaceholder == True:
			# Image processing for training the network. Note the many random
			# distortions applied to the image.

			# Randomly crop a [height, width] section of the image.
			distorted_image = tf.random_crop(image, [height, width, 3])

			# Randomly flip the image horizontally.
			distorted_image = tf.image.random_flip_left_right(distorted_image)

			# Because these operations are not commutative, consider randomizing
			# the order their operation.
			# NOTE: since per_image_standardization zeros the mean and makes
			# the stddev unit, this likely has no effect see tensorflow#1458.
			distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
			distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

			# Subtract off the mean and divide by the variance of the pixels.
			float_image = tf.image.per_image_standardization(distorted_image)

			# Set the shapes of tensors.
			float_image.set_shape([height, width, image.shape[-1]])
		else:
			float_image = inputs_test(image,imSize)

	# Generate a batch of images and labels by building up a queues of examples.
	return float_image

# We changed the files "inputs" in "https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py"
# to generate "inputs_test" used in this file. 
def inputs_test(image,imSize):
    height = imSize
    width = imSize

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image,height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, image.shape[-1]])

    # Generate a batch of images and labels by building up a queue of examples.
    return float_image

