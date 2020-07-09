# Data Science Image Clustering: Project Overview
* Created a tool that clustering image (Accuracy 77%) to help platform classifying image when users upload pictures.
* Built up Autoencoder with Convolutional Neural Network model to extract feature from image.
* Wrote Gradient Descent algorithm to approach minimize loss function by using PyTorch.
* Reduced internal layer dimension by using PCA and t-SNE.
* Used K-Means to cluster.

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, torch, sklearn, matplotlib
**Data:** https://www.kaggle.com/c/ml2020spring-hw9
**Autoencoder Colab:** https://colab.research.google.com/drive/1sHOS6NFIBW5aZGz5RePyexFe28MvaPU6

## Data Set
I used data set of image that be labeled in Kaggle:
* traubX.npy: there are 8500 RGB pictures and the size of picture is 32\*32\*3
* valX.npy: there are 500 RGB pictures and the size of picture is 32\*32\*3
* valY.npy: there are ValX's label. This data set will not use training.    

## Model Building, Dimension Reduction and Clustering
First, I set up autoencoder with CNN model and used PyTorch to pick the best function. Because of image input, I choose covolutional neural network to have less parameters than deep neural network. The autoencoder can help me feature extraction. I thought the model and compute with PyTorch would be effective.
I tried directly to clustering by K-Means, the accuray won't be good. Therefore, I tried to use PCA nad t-SNE to do dimension reduction at first and then clustering by K-Means.
   

## Model performance
 Accuracy = 77%  
The clustering:   
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/Clustering.png)  
The variance of accuracy and error every 10 epochs (total training = 100):    
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/accuracy.png)    
Using the original and reconstrution picture to evaluate the autoencoder:    
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/compare_ori_and_reconstruct.png)    
  
