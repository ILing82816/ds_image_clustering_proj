# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:20:45 2020

@author: USER
"""
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import preprocess as pre
import data as d
import model as m
import train as tr
import predict as p

# check torch.cuda.is_available() , if True set the device to "cuda", if False set device to "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#set parameters: the size of batch, epoch、learning rate
batch_size = 64
epoch =100
lr = 1e-5
path_prefix = 'D:/USA 2020 summer/Machine Learning'

# load data 
print("loading data ...") 
trainX = np.load(os.path.join(path_prefix, '7 Unsupervised/trainX.npy'))
# data preprocess 
trainX_preprocessed = pre.preprocess(trainX)
print(trainX_preprocessed.shape)
# To dataset for dataloader
img_dataset = d.ImageDataset(trainX_preprocessed)
# To batch of tensors
img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True)


# Set up model
model = m.AE()
model = model.to(device) 


# Start training
tr.training(batch_size, epoch, lr, img_dataloader, model, device)


# load model
print('\nload model ...')
model.load_state_dict(torch.load(os.path.join(path_prefix, 'ds_image_clustering_proj/model_state_dict/last_checkpoint.pth')))


#predict
latents = p.inference(X=trainX, model=model, batch_size=256)
pred, X_embedded = p.predict(latents)


# save to csv
print("save csv ...")
p.save_prediction(pred, 'prediction.csv')
# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
p.save_prediction(p.invert(pred), 'prediction_invert.csv')


print("Finish Predicting")

