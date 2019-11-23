## Facial Keypoint Detection

In this project, I build a facial keypoint detection system. The system consists of a face detector that uses Haar Cascades and a Convolutional Neural Network (CNN) that predict the facial keypoints in the detected faces. The facial keypoint detection system takes in any image with faces and predicts the location of 68 distinguishing keypoints on each face.

###### Details on training are covered on jupyter notebook (3.) and implementation of models is at model.py

Some of my output from my Facial Keypoint Detection system:</br>
**NaimishNet**
<p float="left">
  <img src="../images_gifs/face-41.png" width="200" />
  <img src="../images_gifs/face-45.png" width="200" /> 
  <img src="../images_gifs/face-43.png" width="200" />
    <img src="images_gifs/face-44.png" width="200" />
</p>

**ResNet18(w Transfer learning)**
<p float="left">
  <img src="../images_gifs/face-46.png" width="150" />
  <img src="../images_gifs/face-47.png" width="160" /> 
  <img src="../images_gifs/face-48.png" width="165" />
    <img src="../images_gifs/face-49.png" width="160" />
</p>

**Green points: Ground Truth </br>
Purple points: Predicted points by my Model**

Demo gifs on Facial Keypoint detection on videos
<p float="left">
    <img src="../images_gifs/riho_1.gif"/>
      <img src="../images_gifs/riho_2_out.gif", width="480"/>

</p>
