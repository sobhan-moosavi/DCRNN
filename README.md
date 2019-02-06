# Driving Style Representation in Convolutional Recurrent Neural Network Models of Driver Identification

This repository contains all the implementations (including proposed models as well as baselines) and sample files for the D-CRNN paper. 

## Feature Engineering 
* __Statistical Feature Matrix__: to generate feature matrix as input for deep models, use ```Create_FeatureMatrix.py```. 
* __Feature Vector V1__: to generate the original feature vector for a trajectory to be used by GBDT model, use ```Create_FeatureVector_V1.py```. 
* __Feature Vector V2__: to generate the modified feature vector for a trajectory to be used by GBDT model, use ```Create_FeatureVector_V2.py```. 

## Models
* __D-CRNN__: this is our proposed model to perform driver prediction based on driving style information. This model combines several important compoenents of deep-neural-network architectures including recurrent, convolutional, and fully connected components. An implementation of this model in Tensorflow can be find [here](https://github.com/sobhan-moosavi/DCRNN/blob/master/DCRNN.py). Following diagram also describes the design of this model: <div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile modified=\&quot;2019-02-03T01:39:12.872Z\&quot; host=\&quot;www.draw.io\&quot; agent=\&quot;Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36\&quot; etag=\&quot;9bPxho-f9Ks2Qngdc51X\&quot; version=\&quot;10.1.6-6\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;c562b6ec-9891-0046-892f-c501f7719b2d\&quot; name=\&quot;Page-1\&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>

* __VRAE__: this is the Variational Recurrent Auto-encoder model, proposed by [Fabius and Amersfoort (2015)](https://arxiv.org/abs/1412.6581). Here we extend the [original implementation](https://github.com/y0ast/Variational-Recurrent-Autoencoder) to use it for driver prediction task by a modified loss function. The implementation of this model in Theano can be find [here](https://github.com/sobhan-moosavi/DCRNN/blob/master/VRAE.py). 

* __GBDT__: this is a Gradient Boosting Decision Tree model which we use it for driver prediction task. A Python imeplementatin of this method based on scikit-learn library is available [here](https://github.com/sobhan-moosavi/DCRNN/blob/master/GBDT.py). 

* __CNN-model__: this is a Convolutional Neural Network model to perform driver prediction task, which is proposed by [Dong et al. (2016)](https://arxiv.org/abs/1607.03611). Imeplementation of this method in Tensorflow can be find [here](https://github.com/sobhan-moosavi/DCRNN/blob/master/CNN_model.py). 

* __RNN-model__: this is a Recurrent Neural Network model to perform driver prediction task, which is proposed by [Dong et al. (2016)](https://arxiv.org/abs/1607.03611). Imeplementation of this method in Tensorflow can be find [here](https://github.com/sobhan-moosavi/DCRNN/blob/master/RNN_model.py). 

## Requirements
* __Python__: You may use either Python 2.7 or 3. 
* __Tensorflow__: For all the deep models, except VRAE, you need Tensorflow (```version >= 1.12.0``` is recommended). 
* __Theano__: To run VRAE model, you need Theano (```version >= ???``` is recommended). 
* __Cuda Library__: You need Cuda Library for tensorflow and theano-based codes; ```version > 9.0.176``` is recommended. 

Note that our models can be run on both CPU and GPU machines. 

## How to Run
__Creating Feature Vector/Matrix__
* Generate Feature Matrix: Use ```python Create_FeatureMatrix.py``` to create feature matrix. This will result in creating two files in __data__ directory, one _npy_ and one _pkl_ file. 
* Generate Feature Vector: This is to generate input for non-deep models, such as GBDT. For the original version of features as described [here](), use ```Create_FeatureVector_V1.py```, and for the modified version as described in [our paper](), use ```Create_FeatureVector_V2.py```. For each data file, we create one _npy_ and one _pkl_ file, using either of scripts. 

__Modeling and Prediction__
* Run Deep Models: You just need to use ```python [MODEL_NAME].py```. Make sure you have all the requirements satisfied. 
* Run GBDT Models: Set the desired input data file in script and use ```python GBDT.py``` to run the model. Use ```--version v1``` for the original feature vector (size 321), or ```--version v2``` for the modified version (size 384). 

## Sample Data
You may find a raw sample file in [data](https://github.com/sobhan-moosavi/DCRNN/tree/master/data) directory. In this file we have 5 drivers, and 10 random trajectories for each driver. The format of this file is described as follows: 

| Attribute | Description |
|:---------:|-------------|
|Driver| Indicates driver id, which is a string. |
|ID| Indicates trajectory id, which is a string. |
|Time| An integer which indicates the timestep for a datapoint of a trajectory. |
|Lat| Shows the latitude value of GPS coordinate. |
|Lon| Shows the longitude value of GPS coordinate. |
|Speed| Shows the ground velocity of the vehicle as reported by OBD-II port. |
|Acceleration| Shows the rate of change of ground velocity or speed. |
|RPM| Shows the round per minute, as reported by OBD-II port. |
|Heading| Shows the bearing of the vehicle, which is a value between 0 and 359. |
|AccelX| Shows the acceleration sensor reading along with X-axis. |
|AccelY| Shows the acceleration sensor reading along with Y-axis. |
|AccelZ| Shows the acceleration sensor reading along with Z-axis. |


## Sample Results
Following is the result of different models on a random sample set of 50 drivers, with 50 trajectories for each driver. For deep models, we report accuracy on both segment as well as trajectory, see the [paper](#) for details. 

| Model | Accuracy--Segment | Accuracy--Trajectory |
|:-----:|:-----------------:|:--------------------:|
| GBDT-V1| -- | 25.13% |
| GBDT-V2| -- | 49.24% |
| CNN-model | 42.83% | 59.90% |
| RNN-model | 55.45% | 70.73% |
| VRAE | 51.28% | 63.67% |
| D-CRNN | __62.94%__ | __78.00%__ |


## Acknowledgments 
* Sobhan Moosavi, Pravar D. Mahajan, Eric Fosler-Lussier, Colleen Saunders-Chukwu, and Rajiv Ramnath; _"Driving Style Representation in Convolutional Recurrent Neural Network Models of Driver Identification"_, 2019 
