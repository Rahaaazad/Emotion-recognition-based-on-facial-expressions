# Emotion recognition based on facial expressions

This project includes a coding project that shows how to use human faces to recognize their emotions in MATLAB.

# Table of Content
- [Introduction](#1-introduction)
- [Dataset](#2-Dataset)
- [How to run](#3-How_to_run) 
- [Result](#4-Result)
- [References](#5-References)

# Introduction <a name="1-introduction"></a>
Human communication has two main aspects; verbal (auditory) and non-verbal (visual). Facial expression, body movement and physiological reactions are the basic units of non-verbal communication. [[1](#Saudagare)]
Facial expressions play an important role in recognition of emotions and are used in the process of non-verbal communication, as well as to identify people. [[2](#Tarnowski)]. 

The purpose of this project in the first step is to create a strong classifier and validate it on test images, and in the 
next step, recording the live-image and applying the classifier to it, so that different facial expression can be tried on it.

# Dataset <a name="2-Dataset"></a>
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The training set consists of 28,709 examples. The first test set consists of 3,589 examples. The final test set consists of another 3,589 examples.

Note: Only a number of images that have clearer features have been selected and pre-processed that you can find them in folder `TrainData` and `TestData` .
# How to run<a name="3-How_to_run"></a>
1. get data
2. run file `FeatureExtraction.m` to reduce the amount of redundant data from the data set.
3. run file  `TrainModel.m`. Training datasets are fed to machine learning algorithms to teach them how to make predictions.
4. use file `TestModel_ImageSet.m` to evaluate the performance and progress of your algorithms training and adjust or optimize it for improved results.
5. use file  `TestModel_LiveImage.m` to evaluate the performance of live images.

# Result <a name="4-introduction"></a>
Accuracy of Test dataset : ~87%

Confusion matrix :

![untitled1](https://user-images.githubusercontent.com/113347545/196212177-978f4e99-6464-4c6a-a8ee-cd4763f7342b.jpg)

(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
# References <a name="5-References"></a>
[1] <a name="Saudagare"></a>Saudagare, P. V., & Chaudhari, D. S. (n.d.). _International Journal of Soft Computing and engineering ... - citeseerx_. Retrieved September 18, 2022, from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.683.3991&rep=rep1&type=pdf


[2] <a name="Tarnowski"></a> Tarnowski, P., Kołodziej, M., Majkowski, A., & Rak, R. J. (2017, June 9). _Emotion recognition using facial expressions_. Procedia Computer Science. Retrieved September 18, 2022. https://www.sciencedirect.com/science/article/pii/S1877050917305264

    
