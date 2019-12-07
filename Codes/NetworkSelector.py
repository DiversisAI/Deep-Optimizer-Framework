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

def NetworkConfig(netType):
    if netType == 1:
	    layerOutDimSize = [16, 32, 64, 128, 256, 64, 10]
	    kernelSize      = [[3,3],[3,3],[3,3],[3,3],[3,3],[1],[1]]
	    strideSize      = [2,2,2,2,2,1,1]
	    poolKernelSize  = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
	    poolSize        = [1,1,1,1,1,1,1]
	    dropoutInput    = 1.0
	    dropoutPool     = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    elif netType == 2:
        layerOutDimSize = [16,16,32,32,64,64,64,128,128,128,128,128,128,128,10]
        kernelSize      = [[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[1],[1]]
        strideSize      = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        poolKernelSize  = [[2,2],[2,2],[2,2],[2, 2],[2, 2],[2, 2],[2,2],[2,2],[2,2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2]]
        poolSize        = [1,2,1,2,1,1,2,1,1,2,1,1,2,1,1]
        dropoutInput    = 1.0
        dropoutPool     = [0.3,1.0,0.4,1.0,0.4,0.4,1.0,0.4,0.4,1.0,0.4,0.4,0.5,0.5,1.0]
    elif netType == 3:
	    layerOutDimSize = [64, 128, 256, 512, 1024,256,10]
	    kernelSize      = [[3,3],[3,3],[3,3],[3,3],[3,3],[1],[1]]
	    strideSize      = [2,2,2,2,2,1,1]
	    poolKernelSize  = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
	    poolSize        = [1,1,1,1,1,1,1]
	    dropoutInput    = 1.0
	    dropoutPool     = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]    
    elif netType == 4:
        layerOutDimSize = [16,16,16,16,32,32,32,32,64,64,64,64,128,128,128,128,256,256,256,256,256,10]
        kernelSize      = [[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[3, 3],[1],[1]]
        strideSize      = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        poolKernelSize  = [[2,2],[2,2],[2,2],[2,2],[2,2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2,2],[2,2],[2,2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2]]
        poolSize        = [1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1]
        dropoutInput    = 1.0
        dropoutPool     = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    elif netType == 5:
	    layerOutDimSize = [512,128,64,32,10]
	    kernelSize      = [[1],[1],[1],[1],[1]]
	    strideSize      = [1,1,1,1,1]
	    poolKernelSize  = [[3,3],[3,3],[3,3],[3,3],[3,3]]
	    poolSize        = [1,1,1,1,1]
	    dropoutInput    = 1.0
	    dropoutPool     = [1.0,1.0,1.0,1.0,1.0]
    elif netType == 6:
	    layerOutDimSize = [128,128,128,128,128,128,128,128,128,128,128,128,10]
	    kernelSize      = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
	    strideSize      = [1,1,1,1,1,1,1,1,1,1,1,1,1]
	    poolKernelSize  = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
	    poolSize        = [1,1,1,1,1,1,1,1,1,1,1,1,1]
	    dropoutInput    = 1.0
	    dropoutPool     = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    else:
	    print("Network Type is not selected properly")
    
    return layerOutDimSize,kernelSize,strideSize,poolKernelSize,poolSize,dropoutInput,dropoutPool
