


#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
 

 - Use the simulator to collect data of good driving behavior  
 - Build, a convolution neural network in Keras that predicts steering angles from images 
 - Train and validate the model with a training 
 - Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

 - model.py containing the script to create and train the model 
 - drive.py for driving the car in autonomous mode 
 - model.h5 containing a trained convolution neural network   
 - writeup_report.md

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first approach was to construct an architecture that could make the car drive at least a little bit with the provided data by the resource files. I started with a really simple network and evolved to LeNet, then I kept adding augmentation techniques and pre process the images to verify what could make the solution better. 

I started using only the center camera and then all the three cameras. I changed the brightness of some the images and flipped all the images to create more data. All those strategies were good. 

I also tried to add random shadows to the images, but it didn't have a positive impact to the model, so I removed this idea.

After the augmentation and the pre process were good enough, I changed to the architecture by Nvidia. I car then was able to drive a complete lap, but almost getting of the road some times. 

I tried to teach the car how to drive in specific spots that is was seeing bad results, but this was a bad approach. Teach the car how to drive in a more broad way is a better way create training data.

One final modification that resulted in a big change in the way the car drives was to change the number of epochs from three to just one. I believe that it prevented the model to overfit. One curious aspect is that the model didn't need a really small loss to drive correctly. Actually the correctness of the drive was not very connected to the loss of the model. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

This is the final architecture of my model:

 1. A input layer
 2. A lambda layer to normalize the image and apply zero mean. 
 3. A cropping layer to remove the bottom of the image and the top part of the image as it contains useless data. 
 4. A convolution 2D layer with shape of (24, 5 , 5), sub sampling of (2, 2)
 5. An activation layer using relu
 6. A convolution 2D layer with shape of (36, 5, 5), sub sampling of (2, 2)
 7. An activation layer using relu
 8. A convolution 2D layer with shape of (48, 5, 5), sub sampling of (2, 2)
 9. An activation layer using relu
 10. A convolution 2D layer with shape of (64, 5, 5), sub sampling of (2, 2)
 11. An activation layer using relu
 12. A convolution 2D layer with shape of (64, 5, 5), sub sampling of (2, 2)
 13. An activation layer using relu
 14. A flatten layer
 15. A dropout layer with 0.5
 16. A Dense layer with 100
 17. A Dense layer with 50
 18. A Dense layer with 10
 19. A dropout layer with 0.5
 20. A Dense layer with 1

####3. Creation of the Training Set & Training Process

The first three laps were driven in the normal direction of the road and in the center: 

![enter image description here](https://lh3.googleusercontent.com/-DewiaZBv5uM/WZHeuAqaBuI/AAAAAAAAAE4/aWgAditAeksafUQsHo4dCdCHKXBD2zLjQCLcBGAs/s0/center_2017_08_13_17_17_17_775.jpg "center_2017_08_13_17_17_17_775.jpg")

I then recorded the vehicle driving backwards so it could generalize better.

I recorded some turns in the curves so the car could learn not only to drive straight, but also to make curves. I was important to record it in the normal way and backwards. 

![enter image description here](https://lh3.googleusercontent.com/-adLjOX6tuuI/WZHf-d8_6sI/AAAAAAAAAFQ/vQYAKFuSJZIEj7URFbSAC18H7IdS8lLQwCLcBGAs/s0/center_2017_08_13_23_21_43_715.jpg "center_2017_08_13_23_21_43_715.jpg")

It was important to give focus in the curves, instead of just record complete laps,  so the network doesn't waste time training over and over something that it already knows. 

I recorded a lap in the second road. The car got better making curves, but once it hit a yellow line it never left, probably because it learnt to follow the middle line of the road. 

Maybe a good train recovering from the side of the road and the train in the second road could make the model better. The recovering train could help the car to stay out of the side lines.

To augment the data set, I also flipped images and angles thinking that this would help the model to generalize better. Change the brightness was also very good so the model doesn't associate a behavior with the brightness of the image.

After the collection process, I had a large number of data points. But I only used the parts that I observed that were helping the model to drive better. I kept changing the pieces of data until I get the right model. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was only one as evidenced by watching the car drive in the road. This result was very surprising for me as I thought that a higher number of epochs could only help the model to get better. I used an adam optimizer so that manually training the learning rate wasn't necessary.