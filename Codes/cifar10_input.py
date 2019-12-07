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

import numpy as np
import os
import pickle

trainDataSize = 50000
testDataSize = 10000
classNum = 10
inputSize = 3072

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def get_data(dataDirectory):
	dataMatrix = np.zeros((inputSize,trainDataSize))
	labelMatrix = np.zeros((classNum,trainDataSize))
	dataNames = np.zeros((trainDataSize)).astype("string")
	for batchIndx in range(5):
		file = dataDirectory + 'data_batch_%d'%(batchIndx+1)
		absFile = os.path.abspath(file)
		dict = unpickle(absFile)
		data = (np.asarray(dict[b'data'].T).astype("uint8") / 255.0).astype("float32")
		label = np.asarray(dict[b'labels'])
		for i in range(testDataSize):
			labelMatrix[label[i],batchIndx*testDataSize + i] = 1
		dataMatrix[:,batchIndx*testDataSize:(batchIndx+1)*testDataSize] = data
		names = np.asarray(dict[b'filenames'])
		dataNames[batchIndx*testDataSize:(batchIndx+1)*testDataSize] = names
	# Reshape for RGB
	dataMatrix = dataMatrix.reshape(3,32,32,trainDataSize).transpose([1, 2, 0, 3])
	return dataMatrix,labelMatrix,dataNames

def get_data_test(dataDirectory):
	labelMatrix = np.zeros((classNum,testDataSize))
	file = dataDirectory + 'test_batch'
	absFile = os.path.abspath(file)
	dict = unpickle(absFile)
	dataMatrix = (np.asarray(dict[b'data'].T).astype("uint8") / 255.0).astype("float32")
	label = np.asarray(dict[b'labels'])
	for i in range(testDataSize):		
		labelMatrix[label[i],i] = 1
	dataNames = np.asarray(dict[b'filenames'])
	# Reshape for RGB
	dataMatrix = dataMatrix.reshape(3,32,32,testDataSize).transpose([1, 2, 0, 3])
	return dataMatrix,labelMatrix,dataNames
