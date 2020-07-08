https://colab.research.google.com/drive/1sHOS6NFIBW5aZGz5RePyexFe28MvaPU6
# Data Science Image Clustering: Project Overview
* Created a tool that clustering image (Accuracy 77%) to help platform classifying image when users upload pictures.
* Optimized Autoencoder with Convolutional Neural Network to reach the best model.
* Reduced dimension by using PCA and t-SNE and Clustering by K-Means.

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, torch, sklearn, matplotlib
**Data:** https://www.kaggle.com/c/ml2020spring-hw9
**Autoencoder Colab:** https://colab.research.google.com/drive/1sHOS6NFIBW5aZGz5RePyexFe28MvaPU6

## Data Set
I used data set of image that be labeled in Kaggle:
* traubX.npy: there are 8500 RGB pictures and the size of picture is 32*32*3
* valX.npy: there are 500 RGB pictures and the size of picture is 32*32*3
* valY.npy: there are ValX's label. This data set will not use training.    

## Model Building and Clustering
I use PyTorch to Build an Autoencoder with convolutional neural network and evaluated it using Mean Square Error. I chose MSE because it is relatively easy to interpret.  
   

## Model performance
 * **Random Forest:** Accuracy = 77%  
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/Clustering.png)   
 
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/accuracy.png)  
 
![alt text](https://github.com/ILing82816/ds_image_clustering_proj/blob/master/Figure/compare_ori_and_reconstruct.png)  
  
