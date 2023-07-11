# Emojify

## Introduction
1. Built &amp; trained a model in Keras for Facial Expression Recognition
2.  Test the model using real time images
3.  Classified each face based on the emotion shown in the facial expression into one of seven categories (0 = Angry, 1 = Happy, 2 = Neutral, 3 = Sad, 4 = Surprise)
4.  Used OpenCV to automatically detect faces in images and draw bounding boxes around them.
5.  Once we have trained, saved, and exported the CNN, we will directly serve the trained model to a web interface and perform real-time facial expression recognition on video and image data.
6.  Technology Used – Python, OpenCV, CNN, Fer2013 Dataset.

The project can be broadly divided into two parts -
1. Build and train a model in Keras for Facial Expression Recognition.
2. Test the model using real time images

      <img width="331" alt="Screenshot 2023-07-11 at 2 24 24 PM" src="https://github.com/AnshDhalla1/Emojify/assets/83045012/0bc58393-9d70-4c55-b776-a4ef067e5e3f">

## Dataset & its Feature

In this project, the dataset used to train the models is FER-2013. The FER-2013 dataset consists of 35887 images, of which 28709 labelled images belong to the training set and the remaining 7178 images belong to the test set. The images in FER-2013 dataset islabeled as one of the seven universal emotions: Happy, Sad, Angry, Surprise, and Neutral.

<img width="379" alt="Screenshot 2023-07-11 at 2 27 36 PM" src="https://github.com/AnshDhalla1/Emojify/assets/83045012/fc54f372-1681-4fd2-b2f7-8631318fe4bc">

## Implementation & Testing 

**CNN Model**
The CNN designed is based on sequential model and is designed to have six
activation layers, of which 4 are convolutional layers and the remaining 2 are fully
controlled layers.

<img width="566" alt="Screenshot 2023-07-11 at 2 31 12 PM" src="https://github.com/AnshDhalla1/Emojify/assets/83045012/3d8626ef-50bb-4fa6-8d1e-602e48d87e68">

<img width="566" alt="Screenshot 2023-07-11 at 2 34 38 PM" src="https://github.com/AnshDhalla1/Emojify/assets/83045012/64273c25-0086-4650-87ed-c5a46be1c5c6">


## Result 
The CNN model designed is set to undergo 25 epochs. When trained gives an accuracy of 49% after the 25th epoch and the maximum efficiency achieved is also 56%

## Future Work
1. To improve the accuracy of our model. In order to match it to the average accuracy of FER dataset.
2. To integrate this model with the application and make a feedback emoji SAAS as they are one of the most fast-growing and common ways of gathering feedback that companies of all sizes are now using. For business development, customer feedback is significant. It helps you to substantially enhance the customer experience, which directly affects financial growth.
