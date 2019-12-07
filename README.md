# Deep-Optimizer-Framework
Deep Optimizer Framework is a new way of training deep networks that improves inference accuracy on test data developed by DIVERSIS Software (www.diversis.com.tr). This new training approach is applicable to various types of deep neural network architectures. We have tested our proposed Deep Optimizer Framework with six different Deep Neural Network Architectures including Convolutional Neural Networks (CNNs) and Fully Connected Networks (FCNs). The details of network architectures can be seen at https://drive.google.com/open?id=1hED76DQcbEhc31zY9HDvnJsFxHrayOBZ.

To use the Deep Optimizer Framework, the users do not need to change anything in their network design. The users design their own network architecture without any structural or optimization limitations. They simply activate the Deep Optimizer Framework to get better test accuracies for their deep learning applications. Any regularizer and any loss function can be used. In fact, Deep Optimizer Framework is invisible to the user, it only changes the training mechanism for better test accuracy. 

For details please wisit www.diversis.com.tr/

For testing the networks please run "EvalNetwork.py" file. The obtained results will be as follows.

	                    Test Accuracy Results (%)    Difference (%)
             Net Type   Classical	DeepOptimizer
                1	     71,8850	  74,9800	      3,0950
                2	     81,5805	  84,7957	      3,2152
                3	     75,0401	  75,7612	      0,7211
                4	     10,016	  83,2933	      73,2773
                5	     55,3986	  56,1799	      0,7813
                6	     51,8530	  53,6158	      1,7628
   
The reported accuracy results are the maximum test accuracies obtained after running for 1000 epoch.   

You can use "TrainNetwork.py" for training the networks defined in "NetworkSelector.py" file. You can select which network type will be trained via "netTypeList" variable, whose default value is all network types, namely "netTypeList = [1,2,3,5,6,7]".
   
The required modules are,

        Python 2.7.12
        Tensorflow 1.8.0
        Numpy 1.14.6
        Matplotlib 2.2.4
        Pickle 2.0
