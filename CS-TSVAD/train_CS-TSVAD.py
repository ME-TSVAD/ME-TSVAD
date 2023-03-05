import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import re
import random
from tqdm import tqdm
import torch.optim as optim
from torch.optim import Adam
from torch.nn import Parameter, Module, Sigmoid
from torchvision import models
from torchsummary import summary


os.environ["CUDA_VISIBLE_DEVICES"]="0"


'''
### input target speaker embeddings + acustic speaker embediings [512D+512D = 1024D]
input_feat_data = np.load("input_feat_data.npy") 
'''

### set scale = 0.1, 0.3, 0.5, 1, 3 (sec)
scale = 1

### sample point in each utterance
sample_point = int(6/scale)



### load calculated cosine similarity
with open("cos_sim_s1_T.pkl",'rb') as tf:
    s1_T_in = pickle.load(tf)

with open("cos_sim_s2_T.pkl",'rb') as tf:
    s2_T_in = pickle.load(tf)


with open("cos_sim_s3_T.pkl",'rb') as tf:
    s3_T_in = pickle.load(tf)


with open("cos_sim_s4_T.pkl",'rb') as tf:
    s4_T_in = pickle.load(tf)


with open("cos_sim_s5_T.pkl",'rb') as tf:
    s5_T_in = pickle.load(tf)
    


    
cos_sim_55 = []
for i in range(4000):
    cos_sim_55.append( s1_T_in[i] +  s2_T_in[i] +  s3_T_in[i] + s4_T_in[i]+  s5_T_in[i] )
    
    
    
cos_sim_list_T = []  # for 0,2,4 (if scale= 1s)
cos_sim_list_F = []  # for 1,3,5
for i in range(len(cos_sim_55)):

    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    
    for j in range(len(cos_sim_55[i])):  # 55 cos sim , aggregate to 5
        if cos_sim_55[i][j][1] == 1:
           t1.append(cos_sim_55[i][j][0])
        if cos_sim_55[i][j][1] == 2:
           t2.append(cos_sim_55[i][j][0])
        if cos_sim_55[i][j][1] == 3:
           t3.append(cos_sim_55[i][j][0])
        if cos_sim_55[i][j][1] == 4:
           t4.append(cos_sim_55[i][j][0])
        if cos_sim_55[i][j][1] == 5:
           t5.append(cos_sim_55[i][j][0])

    m1 = np.mean(np.array(t1),axis = 0)     
    m2 = np.mean(np.array(t2),axis = 0)     
    m3 = np.mean(np.array(t3),axis = 0)     
    m4 = np.mean(np.array(t4),axis = 0)     
    m5 = np.mean(np.array(t5),axis = 0)     
    
    for frame_idx in range(0,sample_point ,2):  # frame idx = 0,2,4
        cos_sim_T = [m1[frame_idx],m2[frame_idx],m3[frame_idx],m4[frame_idx],m5[frame_idx]] 
        cos_sim_list_T.append(cos_sim_T)
   
    for frame_idx in range(1,sample_point ,2):  # frame idx = 1,3,5

        cos_sim_F = [m1[frame_idx],m2[frame_idx],m3[frame_idx],m4[frame_idx],m5[frame_idx]] 
        cos_sim_list_F.append(cos_sim_F)

print("total frame =")
print(len(cos_sim_list_T))  # 4000*6/(scale*2)
print(len(cos_sim_list_F))  # 4000*6/(scale*2)

input_data = np.array(cos_sim_list_T+cos_sim_list_F)

print("input data shape 1 : [cos_sim]")
print(input_data.shape)

print("input data shape 2 : [select feature]")
print(input_feat_data.shape)


# label = 4000 utterance * sample_points /2  (half for True and half for False)

la1 = np.ones(4000*sample_point/2) # half are True
la0 = np.zeros(4000*sample_point/2) # half are False

# convert to one hot
input_label = np.eye(2)[np.concatenate((la1,la0)).astype(int)]


#print("input label shape")
#print(input_label.shape)


###################################
### trainable binary classifier ###
################################### 
        
batch_size = 50



class Network(nn.Module):

    def __init__(self):
        super().__init__()

        
        ### structure of Net4 (7-dimensional cosine similarity set)
        self.fcm1 = nn.Linear(7,4)   
        self.bm1 = nn.BatchNorm1d(4)
        self.fcm2 = nn.Linear(4,2)
        self.bm2 = nn.BatchNorm1d(2)


        
    


    def forward(self,x1,x2):

        
        ### add mean and variance (5+2 = 7 dimensional)
        x1 =  torch.cat((x1, torch.stack(torch.var_mean(x1, dim = 1),dim = 1)  ), dim=1)
        ### x1 : 7-dimensional cosine similarity set

        
        ### no need for Net 1 and Net2 
        ### (We donlt use acoustic feature embeddings and target feature embeddings in CS-TSVAD)

        ### Net4 forward
        z = self.fcm1(x1) #  directly input 7-dimensional cosine similarity set
        z = self.bm1(z)
        z = self.fcm2(z)
        z = self.bm2(z)
        z = F.softmax(z,dim = 1)
        return z





net = Network()
net.cuda()


# calculate total parameter
summary(net, [(5,), (1024,)]) ### 5D+512D+512D

num_params_total = sum(param.numel() for param in net.parameters())
print("total parameter")
print(num_params_total)

# trainable parameter
num_params_trainable = sum(param.numel() for param in net.parameters() if param.requires_grad)
print("trainable parameter")
print(num_params_trainable)




### ConcatDataset
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



### train dataset loader
train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 [[input_data[i], input_label[i]] for i in range(len(input_label))],
                 [[input_feat_data[i], input_label[i]] for i in range(len(input_label))]
             ),
             batch_size= batch_size, shuffle=True)            
             

'''
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Cuda Device Available")
  print("Name of the Cuda Device: ", torch.cuda.get_device_name())
  print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
'''
    

### train Network
def train(model, x1,x2, y, optimizer, criterion):
    model.zero_grad()
    output = model(x1,x2)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss, output



EPOCHS = 1000 #(500~5000)
acc = 0
criterion = nn.CrossEntropyLoss() ### crossentropy loss
optm = Adam(net.parameters(), lr = 1e-5 , weight_decay = 1e-5) ### Adam optimizer

#scheduler = optim.lr_scheduler.StepLR(optm, step_size=200, gamma=0.1)  
for epoch in range(EPOCHS):


    epoch_loss = 0
    correct = 0
    net.train()
    for i, (batch1, batch2) in enumerate(train_loader):
        
        x1 = batch1[0].to(device).float()
        x2 = batch2[0].to(device).float()
        y_train = batch1[1].to(device).float()
        loss, predictions = train(net,x1,x2,y_train, optm, criterion)
        cnt = len(predictions) - int(sum(abs(torch.argmax(predictions, dim=1) - torch.argmax(y_train, dim=1))))
        correct += cnt
        acc = (correct/len(input_label))
        epoch_loss+=loss
    print('Epoch {} Accuracy : {}'.format(epoch+1, acc*100))
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))


### save the model
torch.save(net.state_dict(), 'model_CS-TSVAD.pth')


