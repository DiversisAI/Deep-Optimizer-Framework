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
import NetworkSelector as NetSelect
import Inference_NetworkConfiguration as Net
import cifar10_input as DataSet

DataSetFileName = "../Dataset_CIFAR/cifar-10-batches-py/"

fileName     = []
netType      = []
augment      = []
resultsTrain = []
resultsTest  = []
# Determine network to be tested 
netType_list = [1,2,3,4,5,6]
augment_list = [False,False,False,False,False,False]
for i in range(len(netType_list)):
    fileName.append('../PreTrainedNetworkStructures/NetType-%d_Classical-0'%netType_list[i])
    netType.append(netType_list[i])
    augment.append(augment_list[i])

    fileName.append('../PreTrainedNetworkStructures/NetType-%d_DeepOptimizer-0'%netType_list[i])
    netType.append(netType_list[i])
    augment.append(augment_list[i])

# Construct selected network
InputSize =[32,32,3]
testBatchSize = 64
dataTest,labelsTest,namesTest = DataSet.get_data_test(DataSetFileName)
dataTrain,labelsTrain,namesTrain = DataSet.get_data(DataSetFileName)
frameBufferForTest = np.zeros([testBatchSize, InputSize[0], InputSize[1], InputSize[2]],dtype=np.float32)
for k in range(len(fileName)):
    tf.reset_default_graph()
    if augment[k] == False:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 24
    layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,dropoutInput,dropoutPool = NetSelect.NetworkConfig(netType[k])
    # Construct Network
    netOut,accuracyOp,placeHolders = Net.ConstructInferenceNetwork(InputSize,testBatchSize,layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,IMAGE_SIZE)    
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("\n\n\n")
        print(fileName[k])
        saver.restore(sess, fileName[k])
        print('Network loaded')
        print("Testing ...")
        accuracyTrain = Net.GetTestAccuracy(sess,accuracyOp,dataTrain,labelsTrain,testBatchSize,frameBufferForTest,placeHolders['inputFramePlaceHolder'],placeHolders['inputDistortionPlaceholder'],placeHolders['labelPlaceHolder'],placeHolders['dropoutInputPlaceHolder'],placeHolders['dropoutPoolPlaceHolder'])
        accuracyTest = Net.GetTestAccuracy(sess,accuracyOp,dataTest,labelsTest,testBatchSize,frameBufferForTest,placeHolders['inputFramePlaceHolder'],placeHolders['inputDistortionPlaceholder'],placeHolders['labelPlaceHolder'],placeHolders['dropoutInputPlaceHolder'],placeHolders['dropoutPoolPlaceHolder'])
        print("Test accuracy / Train accuracy: %f / %f"%(accuracyTest,accuracyTrain))
        resultsTrain.append(accuracyTrain)
        resultsTest.append(accuracyTest)
        
print("**************** RESULTS ****************")
for i in range(len(resultsTrain)):
    print(fileName[i])
    print("Test accuracy / Train accuracy: %f / %f\n\n"%(resultsTest[i],resultsTrain[i]))

