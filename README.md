# **Behavioral Cloning Project** 

## Author : Sandeep Patil

### Objective of this project is to clone driver's steering behavior using camera images as source. Here we will train neural network to learn the expected steering angle based on given situation. 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior [simulator](https://s3.amazonaws.com/video.udacity-data.com/topher/2017/February/5894ea69_beta-simulator-windows/beta-simulator-windows.zip)
* Build, a convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one.

[//]: # (Image References)

[sample_images]: ./sample_images/original_images.png "sample images"
[flip_cropped]: ./sample_images/flip_and_crop_images.png "flip cropped"
[loss]: ./sample_images/training_loss.png "flip cropped"
[gif_image]: ./sample_images/demo.gif "demo gif"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![demo image][gif_image]


### The data collection 

1. We need to collect training data using simulator. 
 * While we drive in simulator we can record our driving. During recording the image frames and its control details get captured in driving_log.csv.
 * The driving_log.csv file has following details : center camera image path, left camera image path ,right camera image path, steering, throttle, brake and	speed
2. Following points should be consider during collecting data. 
   * Recording two or three laps of center lane driving
   * Recording one lap of recovery driving from the sides
   * Recording one lap focusing on driving smoothly around curves
   * Recording one lap in other direction.
3. We can also use the data provided on udacity : [ data.zip ](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)


### The Data exploration and augmentation and processing. 

Due to system constraint I was not able to record images properly. So I have used provided data to train network. Following are sample images.

![sample images][sample_images] 

#### Data Augmentation
I used following augmentation technique to get ample amount of data. 

1. Use images from all 3 cameras.
  * Adjust steering for left and right images by adding and subtracting 0.15 respectively from steering angle of center image. This would help to mimic recovering from side of road.
* As provided images are of mostly of clock wise turns so we need to add images with counter clock wise turns.
   * For this I flipped images and changed sign of steering angle (-1.0 * original angle)

#### Preprocessing

1. As area above horizon and portion of car bonnet is not of interest. I have cropped image as follow.
   * 65 pixel from top
   * 20 pixel from bottom
2. Normalizing images :
   * I have normalized image to avoid saturation of network.
   
The cropping and normalization are done in as part of model. The preprocessed images should look like below images.

![flipped and cropped][flip_cropped]
   
### The neural network architecture.

* I used neural network architecture of [NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf) which address same problem.
* Added drop out layers after each fully connected layer. 
* I have used Keras apis to build and run model. My final model consisted of the following layers:

| Layer         		|     Description	        					| Learning parameter |
|:---------------------:|:---------------------------------------------:| :-----------------:|
| Input         		| 160x320x3                   					|                    |
| Normalization 		| X/255.0 - 0.5                                 |                    |
| Cropping         		| output : 75 x 320 x 3        					|                    |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 24@36x158 	|      1,824         |
| ELU activation        |												|                    |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 36@16x77 	|     21,636         |  
| ELU activation        |												|                    |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 48@6x37 	|     43,248         |
| ELU activation        |												|                    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 64@4x35    |     27,712         |
| ELU activation        |												|                    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 64@2x33    |     36,928         |
| ELU activation        |												|                    |
| Flatten  	      		| outputs 4224				 					|                    |
| Dropout				| 50%											|                    | 
| Fully connected(Dense)| Outputs 100  									|    422,500         |
| ELU					|												|                    |
| Dropout				| 50%											|                    | 
| Fully connected(Dense)| Outputs 50  									|      5,050         |
| ELU					|												|                    | 
| Dropout				| 20%											|                    |
| Fully connected(Dense)| Outputs 10 logits 							|        510         | 
| ELU					|												|                    | 
| Fully connected(Dense)| Outputs 1         							|         11         | 
| MSE				    | Loss using mean squared error.                |                    |
|                       |       Total Learnable parameters              |    559,419         |     

The additional layers apart from NVidia models which added are as follow.
* Cropping images before feeding to network.
* Drop outs after each fully connected layer to overcome overfitting.
* ELU activation layer to activate neurons and to take care of vanishing gradient problem.

### Training and testing.
The data augmentation and model definition in maintained in model.py .
1. I trained models on GPUs.
2. I trained it for 5 Epochs.
3. Training examples per epoch : 8000 * 6 = 48,000 total images.
4. Training and validation ratio : 80:20


Following is training loss per epoch.

![loss][loss]

At end program saves model in model.h5 file, which is used to give steering commands to simulator.

### Testing on simulator.
The simulator can be run in autonomous mode by giving following command.

    
    > python drive.py model.h5
    
### Sample video 
The sample run on simulator for 1 lap is recorded in output_video.mp4 .
    
### Problems faced during :
1. The drive.py feeds image in RGB format and I had not put this provision earlier. So though training loss was low most of time the car in simulator was going off track. 
2. I rectified this problem by adding code at preprocessing stage to convert image to RGB format and retrained my model again.

## How to set up project in your machine. 

1. Install miniconda
2. Get this repository on your machine 
    ```
    > git clone https://github.com/sandysap22/clone_steering_behavior.git
    ```
3. Got to setup_details folder and hit conda command to get all required software. Note that if you are on windows then need to rename windows_meta.yml to meta.yml
    ```
    > conda env create -f environment.yml
    ```
4. Activate your newly created environment.
    ```
    > source activate carnd-term1
    ```
5. Download  [data.zip](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip) and extract it.
6. Train model 
    ```
    > python model.py
    ```
