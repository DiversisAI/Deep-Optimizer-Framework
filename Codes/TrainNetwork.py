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
import Utility as util
import math
import pickle

netTypeList = [1,2,3,4,5,6]
for netIndx in range(len(netTypeList)):
    tf.reset_default_graph()
    netName         = 'NN'
    netType         = netTypeList[netIndx]
    augment         = False
    batchSize       = 64
    momentum        = 0.9
    weightDecay     = 5e-4
    DataSetFileName = "../Dataset_CIFAR/cifar-10-batches-py/"
    ResultsDirName  = "../Results/"
    TestingEpoch    = 1
    maxEpoch        = 1000
    # Read CIFAR10 dataset train and test data
    print("Reading dataset ...")
    dataTrain,labelsTrain,namesTrain = DataSet.get_data(DataSetFileName)
    dataTest,labelsTest,namesTest = DataSet.get_data_test(DataSetFileName)
    InputSize = list(dataTrain.shape[0:-1])
    MaxFileNum = dataTrain.shape[-1]
    if augment == False:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 24
    TrainEvalEpochNum = np.round(MaxFileNum/batchSize)
    MaxTrialNum = maxEpoch*TrainEvalEpochNum
    print("Reading dataset completed")
    # Get network configuration parameters
    layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,dropoutInput,dropoutPool = NetSelect.NetworkConfig(netType)
    # Construct Network
    netOut,accuracyOp,placeHolders = Net.ConstructInferenceNetwork(InputSize,batchSize,layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,IMAGE_SIZE)  
    # Construct Optimizer
    train_step,totLossOp,l2LossOp,learningRatePlaceHolder = Net.ConstructOptimizer(netOut, placeHolders['labelPlaceHolder'], momentum,weightDecay)  
    # Create buffers for training and testing 
    frameBufferForTrain = np.zeros([batchSize, InputSize[0], InputSize[1], InputSize[2]],dtype=np.float32)
    frameBufferForTest  = np.zeros([batchSize, InputSize[0], InputSize[1], InputSize[2]],dtype=np.float32)
    # Load the previously saved Network Weights into default graph
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    netFileName = 'NetworkStructures/' + netName + '_Type-%d'%netType + '_Augment-%d'%augment
    # Open figures to show train and test accuracies
    liD,ax,fig = util.OpenFigure("Epoch","Accuracy","Train Accuracy")
    xList = []
    liDTest,axTest,figTest = util.OpenFigure("Epoch","Accuracy","Test Accuracy")
    trainAccuracyResults = []
    testAccuracyResults = []
    with tf.Session() as sess:
        try:
            # Initialize values with saved data
            saver.restore(sess, netFileName+'-0')
            print(netFileName+'-0')
            print('Network Initialized with saved data')
        except:
            sess.run(tf.global_variables_initializer())
            print("Network Initialized with random variables")

        dataList = np.arange(MaxFileNum)
        maxTestAccuracy = 0.0
        totTrainAccuracy = 0.0
        trainAccuracySumCnt = 0
        for j in range(MaxTrialNum):
            np.random.shuffle(dataList)
            # Adapt Learning Rate
            if j < 300*TrainEvalEpochNum:
                learningRate = 1e-2
            elif j < 600*TrainEvalEpochNum:
                learningRate = 1e-3
            elif j < 800*TrainEvalEpochNum:
                learningRate = 1e-4
            else:
                learningRate = 1e-5
            # Read Training Data
            frameBufferForTrain,dataList,batchLabel = util.ReadBatchData(dataTrain,labelsTrain,batchSize,frameBufferForTrain,dataList,MaxFileNum)
            feed_latent = {placeHolders['inputFramePlaceHolder']:frameBufferForTrain, placeHolders['inputDistortionPlaceholder']:augment, learningRatePlaceHolder:learningRate, placeHolders['labelPlaceHolder']:batchLabel, placeHolders['dropoutInputPlaceHolder']:dropoutInput, placeHolders['dropoutPoolPlaceHolder']:dropoutPool}
            # Perform training
            _, totLoss_Val, trainAccuracy_Val, l2Loss_Val = sess.run([train_step,totLossOp, accuracyOp, l2LossOp],feed_dict = feed_latent)
            if math.isnan(l2Loss_Val) == False:
                # Perform testing
                if j % (TrainEvalEpochNum*TestingEpoch) == 0 and j > 0:
                    lastTrainAccuracy = totTrainAccuracy /  trainAccuracySumCnt
                    totTrainAccuracy = 0.0
                    trainAccuracySumCnt = 0
                    accuracyTest = Net.GetTestAccuracy(sess,accuracyOp,dataTest,labelsTest,batchSize,frameBufferForTest,placeHolders['inputFramePlaceHolder'],placeHolders['inputDistortionPlaceholder'],placeHolders['labelPlaceHolder'],placeHolders['dropoutInputPlaceHolder'],placeHolders['dropoutPoolPlaceHolder'])
                    epoch = int(j / TrainEvalEpochNum)
                    # Save network if test accuracy increased
                    if accuracyTest > maxTestAccuracy:
                        maxTestAccuracy = accuracyTest
                        save_path = saver.save(sess, netFileName, global_step = 0)
                        print("NETWORK SAVED to %s"%save_path)
                    trainAccuracyResults.append(lastTrainAccuracy)
                    testAccuracyResults.append(accuracyTest)
                    xList.append(epoch)
                    # Show accuracy results
                    util.ShowFigure(liD,ax,fig,xList,trainAccuracyResults)
                    util.ShowFigure(liDTest,axTest,figTest,xList,testAccuracyResults)
                    print("*** Results @Epoch %d ***"%epoch)
                    print("Loss           = %f"%totLoss_Val)
                    print("Train accuracy = %f"%lastTrainAccuracy)
                    print("Test accuracy  = %f (Max: %f)"%(accuracyTest,maxTestAccuracy))
                    # Save results
                    outfile = ResultsDirName + netName + '_Type-%d'%netType + '_Augment-%d'%augment + ".pkl"
                    accuracyResults = {'xList':xList,'trainAccuracy':trainAccuracyResults, 'testAccuracy':testAccuracyResults}
                    saveFile = open(outfile,"wb")
                    pickle.dump(accuracyResults,saveFile)
                    saveFile.close()
                totTrainAccuracy += trainAccuracy_Val
                trainAccuracySumCnt += 1                
            else:
                # Reload last valid weights
                # Initialize values with saved data
                saver.restore(sess, netFileName+'-0')
            
