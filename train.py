# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:57:55 2020

@author: USER
"""
import torch
from torch import nn
import torch.optim as optim
import extrafunction as ex

def training(batch_size, n_epoch, lr, train, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable)) 
    
    ex.same_seeds(0)
    
    loss = nn.MSELoss() # define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # set Adam as optimizer
    for epoch in range(n_epoch):
        total_loss = 0
        
        # start training
        model.train()
        for inputs in train:
            inputs = inputs.to(device) 
            
            optimizer.zero_grad() 
            output1, output = model(inputs) 
            batch_loss = loss(output, inputs) # compute training loss
            batch_loss.backward() # gradient
            optimizer.step() 
            
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), 'D:/USA 2020 summer/Machine Learning/ds_image_clustering_proj/model_state_dict/checkpoint_{}.pth'.format(epoch+1))
                
            total_loss += batch_loss.item()
            
        print('\nepoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, total_loss))
        
    torch.save(model.state_dict(), 'D:/USA 2020 summer/Machine Learning/ds_image_clustering_proj/model_state_dict/last_checkpoint.pth')


