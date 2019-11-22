# Computer Vision Nanodegree 
[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

This repository contains project files for Udacity's Computer Vision Nanodegree program which I enrolled on 10 August, 2019.

## Projects

### Facial Keypoint Detection
>[P1_Facial_Keypoints](https://github.com/nz-is/CVND-Projects/tree/master/P1_Facial_Keypoints)

In this project, I build a facial keypoint detection system. The system consists of a face detector that uses Haar Cascades and a Convolutional Neural Network (CNN) that predict the facial keypoints in the detected faces. The facial keypoint detection system takes in any image with faces and predicts the location of 68 distinguishing keypoints on each face.

Some of my output from my Facial Keypoint Detection system:</br>
**NaimishNet**
<p float="left">
  <img src="images_gifs/face-41.png" width="200" />
  <img src="images_gifs/face-45.png" width="200" /> 
  <img src="images_gifs/face-43.png" width="200" />
    <img src="images_gifs/face-44.png" width="200" />
</p>

**ResNet18(w Transfer learning)**
<p float="left">
  <img src="images_gifs/face-46.png" width="150" />
  <img src="images_gifs/face-47.png" width="160" /> 
  <img src="images_gifs/face-48.png" width="165" />
    <img src="images_gifs/face-49.png" width="160" />
</p>

**Green points: Ground Truth </br>
Purple points: Predicted points by my Model**

![](images_gifs/riho_2_out.gif)

### Automatic Image Captioning
>[P2_Image_captioning](https://github.com/nz-is/CVND-Projects/tree/master/P2_Image_Captioning)

In this project, I design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for automatically generating image captions. The network is trained on the Microsoft Common Objects in COntext (MS COCO) dataset. The image captioning model is displayed below.

![Image Captioning Model](images_gifs/cnn_rnn_model.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)
