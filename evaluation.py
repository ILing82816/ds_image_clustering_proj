# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:04 2020

@author: USER
"""
import extrafunction as ex
import numpy as np
import predict as p
import model as m
import data as d
import torch
import matplotlib.pyplot as plt
import preprocess as pre
import glob
from torch.utils.data import DataLoader
import os

path_prefix = 'D:/USA 2020 summer/Machine Learning'
# load data 
print("loading data ...")
trainX = np.load(os.path.join(path_prefix, '7 Unsupervised/trainX.npy')) 
valX = np.load(os.path.join(path_prefix, '7 Unsupervised/valX.npy'))
valY = np.load(os.path.join(path_prefix, '7 Unsupervised/valY.npy'))
#data preprocess 
trainX_preprocessed = pre.preprocess(trainX)
# To dataset for dataloader
img_dataset = d.ImageDataset(trainX_preprocessed)
#  To batch of tensors
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = m.AE()
model = model.to(device) 
# load model
print('\nload model ...')
model.load_state_dict(torch.load(os.path.join(path_prefix, 'ds_image_clustering_proj/model_state_dict/last_checkpoint.pth')))


#predict
latents = p.inference(X=valX, model=model)
pred_baseline, embedded_baseline = p.predict(latents)


#draw clustering
acc_latent = ex.cal_acc(valY, pred_baseline)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
ex.plot_scatter(embedded_baseline, valY, savefig='Figure/Clustering.png')



# compare original and reconstruct picture
plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7,9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)
    #plt.savefig("Figure/original.png")
    

inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
    plt.savefig("Figure/compare_ori_and_reconstruct.png")
    

#draw every checkpoint mean square error and accuracy
checkpoints_list = sorted(glob.glob('D:/USA 2020 summer/Machine Learning/ds_image_clustering_proj/model_state_dict/checkpoint_*.pth'))
points = []
with torch.no_grad():
    for i, checkpoint in enumerate(checkpoints_list):
        print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        err = 0
        n = 0
        for x in img_dataloader:
            x = x.cuda()
            _, rec = model(x)
            err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
            n += x.flatten().size(0)
        print('Reconstruction error (MSE):', err/n)
        latents = p.inference(X=valX, model=model)
        pred, X_embedded = p.predict(latents)
        acc = ex.cal_acc(valY, pred)
        print('Accuracy:', acc)
        points.append((err/n, acc))

ps = list(zip(*points))
plt.figure(figsize=(6,6))
plt.subplot(211, title='Reconstruction error (MSE)').plot(ps[0])
plt.subplot(212, title='Accuracy (val)').plot(ps[1])
plt.savefig("Figure/accuracy.png")
plt.show()

