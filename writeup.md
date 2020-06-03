# **Traffic Sign Recognition** 

## Writeup

<!-- ![Traffic Signs](./results_images/00_traffic_signs.jpg) -->
|<img src="./results_images/00_traffic_signs.jpg" width="500" height="300" align="center"/>
|:--:| 
|*German Signs Image*|


## Build a Traffic Sign Recognition Project

### Deep Learning

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


<!-- [//]: # (Image References)

[image1]: ./results_images/ "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5" -->

[//]: # (Image References)

[img1]:  ./results_images/1_explored_images.png       "Explored_images"
[img2]:  ./results_images/2_labels_histo.png          "Labels Dist."
[img3]:  ./results_images/3_data_augmentation.png     "Image Processing"
[img4]:  ./results_images/4_train_augmented_histo.png "Train Processed Imgs"
[img5]:  ./results_images/5_preprocess.png            "Preprocess"
[img6]:  ./results_images/6_train_accu_loss.png       "Train, Accu., loss"
[img7]:  ./results_images/7_web_images.png            "Web Images"
[img8]:  ./results_images/8_preprocess_web_images.png "Preprocess W_images"
[img9]:  ./results_images/9_web_images_class.png      "W_images classi."
[img10]: ./results_images/10_web_images_predict.png   "W_images predict"

## Project

You could reach the project implementation [Traffic_Sign_Classifier.ipynb](https://github.com/aliasaswad/CarND-Traffic-Sign-Classifier-P3/blob/master/Traffic_Sign_Classifier.ipynb). Below, I will go through the details of the code implementation and I will discuss the results I've got.

### Step 0: Load The Data

In this steps, I used pickle library to load the data. The data consist of images and labels ad a numbers for these images. The labels are used to recognize what the images represent. [signnames.cvs](https://github.com/aliasaswad/CarND-Traffic-Sign-Classifier-P3/blob/master/signnames.csv) is a file that used to map lables to names, for example:

```python
label         name

 0:    'Speed limit (20km/h)',
 1:    'Speed limit (30km/h)',
 2:    'Speed limit (50km/h)',
 3:    'Speed limit (60km/h)',
 4:    'Speed limit (70km/h)',
 5:    'Speed limit (80km/h)',
 6:    'End of speed limit (80km/h)',
 7:    'Speed limit (100km/h)',
 8:    'Speed limit (120km/h)',
 9:    'No passing',
```

### Step 1: Data Set Summary & Exploration

I used the [pandas library](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html) to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set = 43

Here is an exploratory visualization of the dataset that randomly selected from 43 different images.

![alt text][img1]

Also, the below figures show how the labels are distribute

![alt text][img2]

#### Data Augmentation

In order to have more images to train out the model, I used applied image on the images and concatenate those images to the original images. The process called [data augmentation](https://www.techopedia.com/definition/28033/data-augmentation). The new images created from the original data by using image transformation (image processing). Four types of transformation were used, translation, resizing, and rotation.

```python
def image_process(image, process = 0):
    """
    Applying image processing like scale, transform, and translation.
    We need it when the training data is complex and have a few samples.
    """
    # Scale
    if process == 1:
        fx = 1.5; fy = 1.5
        return cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)[0:32, 0:32,:]
    # Translation
    if process == 0: 
        X = 5; Y = 5
        P = np.float32([[1,0,Y],[0,1,X]])
    else:
        angle = 45
        P = cv2.getRotationMatrix2D((16,16),angle,1)
    
    return cv2.warpAffine(image,P,(32,32))
```

A sample from the output images as below

![alt text][img3]

The new transformed images concatenated with the original images and both used to train the model which provided even more distribution as shown below

![alt text][img4]


    
### Step 2: Design and Test a Model Architecture

[Deep learning neural network models learn a mapping from input variables to an output variable](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/). Data preparation involves using techniques such as the normalization and standardization to rescale input and output variables prior to training a neural network model. Different ways of preprocessing are available to improve the image qualities like use gray scale and normalization. Min-max normalizationis one of the most common ways to normalize data. For every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1.


As a first step, I decided to convert the images to grayscale

```pyhton
  # RGB to gray scale
  gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
```
As a last step, I normalized the image data

```pyhton
  # Normalized Data 'x'
  normalized (x) = (x-min(x))/(max(x)-min(x))
```
I decided to generate additional data and concatenate them to the original data set.
Here is an example of a traffic sign image after grayscaling and normalization.

![alt text][img5]


#### Model architecture 

I used [LeNet](http://yann.lecun.com/exdb/lenet/), , convolutional neural networks (Yann LeCun, 1998) for my model architecture. This model was designed for hand written and machine printed character recognition. So, the model could be a good fit for the traffic sign classification. To improve the accuracy for the model to work with traffic signs, I maked the first two convolution layer deeper, also increase the size of the fully-connected layers. In addition two dropout layers were added. The new architecture that I applied makes accuracy above %95. The final model consisted of the following layers:

| Layer         		|     Description	        				| 
|:---------------------:|:-----------------------------------------:| 
| Input         		| 32x32x3 RGB image   						| 
| Convolution 1     	| 1x1 stride, same padding, outputs 28x28x16|
| RELU					|											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 			|
| Convolution 2 	    | 1x1 stride, same padding, outputs 10x10x64|
| RELU					|											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   			|
| Fatten					|										|
| Fully-connected 1		| 1600 										|
| RELU					|											|
| Dropout					|										|
| Fully-connected 2		| 240										|
| RELU					|											|
| Dropout				|											|
| Fully connected 3		| 43										|
| Softmax				| etc.        								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


