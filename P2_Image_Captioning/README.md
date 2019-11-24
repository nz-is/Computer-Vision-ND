## Automatic Image Captioning

In this project, I design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for automatically generating image captions. The network is trained on the Microsoft Common Objects in COntext (MS COCO) dataset. The image captioning model is displayed below.

![Image Captioning Model](../images_gifs/cnn_rnn_model.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)

### Important Notes
```
1. Implementation of Model Architecture (CNN-RNN) is at models.py
2. Traning details are covered on notebook (2.)
3. Examples of running an Inference on images is at (3.)  
```
Some output of my Automatic Image Captioning System:

### To-Dos
1. Running inference on Videos through feature averaging from the Encoder(CNN) 
2. Add Attention Mechanism the architecture 
