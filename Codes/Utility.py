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
import matplotlib.pyplot as plt

def OpenFigure(xlabel,ylabel,title):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	liD, = ax.plot([], [], 'b-')

	# draw and show it
	fig.canvas.draw()
	plt.show(block=False)
	plt.xlabel(xlabel, fontsize=12)
	plt.ylabel(ylabel, fontsize=12)
	plt.title(title, fontsize=12)

	return liD,ax,fig

def ShowFigure(liD,ax,fig,xList,yList):
	liD.set_xdata(xList)
	liD.set_ydata(yList)

	ax.relim()
	ax.autoscale_view(True,True,True)

	fig.canvas.draw()

def ReadBatchData(data,labels,batchSize,frameBufferForTrain,dataList,MaxFileNum):
    batchLabels = np.zeros((batchSize,labels.shape[0]))
    for i in range(batchSize):
        frameBufferForTrain[i,:,:,:] = data[:,:,:,dataList[0]]
        batchLabels[i,:] = labels[:,dataList[0]]
        dataList = np.delete(dataList,0)
    if len(dataList) < (batchSize):
        dataList = np.arange(MaxFileNum)

    return frameBufferForTrain,dataList,batchLabels

