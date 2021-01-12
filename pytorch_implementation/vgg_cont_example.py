import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import ImagenetLoader



class ContinuousPool(nn.Module):
    def __init__(self, in_channels, timesteps=3, type='max', ksize=2, strides=2, padding=1):
        super(ContinuousPool,self).__init__()
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.pool_op = nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=(1, 1))
        
        if type == 'avg':
            self.result_pool_op = nn.AvgPool2d(kernel_size=ksize, stride=strides, padding=padding)
        elif type == 'max':
            self.result_pool_op = nn.MaxPool2d(kernel_size=ksize, stride=strides, padding=padding)
        else:
            raise ValueError('The available pooling types are \'avg\' and \'max\'!')

        self.pool_strength=nn.Parameter(torch.Tensor(np.full((timesteps,1, self.in_channels, 1, 1), 0.8)).float(), requires_grad=True)
        #!!!Maybe limiting the variable values 

        #self.pool_strength =torch.sigmoid(nn.Parameter(torch.Tensor(np.full((timesteps,1, self.in_channels, 1, 1), 0.0)).float(), requires_grad=True) ).cuda()
         
        
        
    def forward(self, input):
        current_shape = input.size()
        
        new_input = input
        pool_strength  = self.pool_strength.repeat((1,current_shape[0], 1, current_shape[2], current_shape[3]))
        
        for i in range(self.timesteps):
            pooled = self.pool_op(new_input)
            diff = pooled - new_input
            new_input = new_input + (pool_strength[i,:,:,:,:]    * diff)
       
        return self.result_pool_op(new_input)


class BN2d(nn.Module):
    def __init__(self, Channels, Thes=2.0  ):
        super(BN2d , self).__init__()
        self.ChannelNum=Channels
        self.beta==nn.Parameter(torch.tensor([0.0]*Channels), requires_grad=True).cuda()
        self.gamma==nn.Parameter(torch.tensor([1.0]*Channels), requires_grad=True).cuda()
        self.beta=self.beta.reshape(1,Channels,1,1)
        self.gamma=self.gamma.reshape(1,Channels,1,1)
        self.Thes=Thes
        
    def forward(self, xorig):
        x=xorig.permute([1,0,2,3])
        x=x.reshape((self.ChannelNum,-1))
        
        Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Mean=Mean.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        Var=Var.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        
        eps=1e-20
        normalized= (xorig-Mean)/torch.sqrt(Var+eps)     
        Selected= ((normalized<self.Thes) * (normalized>-self.Thes)).float()
        #masked mean
        Mean=torch.sum(xorig*Selected, dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        Mean=Mean.reshape((1,self.ChannelNum,1,1))
        Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        Diff=(xorig - Mean)**2
        Var= torch.sum(Diff*Selected , dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        Var=Var.reshape((1,self.ChannelNum,1,1))
        Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        """
        #channelwise mean and variance
        Mean=torch.zeros(self.ChannelNum).cuda()
        Var=torch.zeros(self.ChannelNum).cuda()
        for i in range(self.ChannelNum):
               Channel=xorig[:,i,:,:]
               Selected[:,i,:,:]
               prunedx=Channel[Selected[:,i,:,:]]
               Mean[i]=torch.mean(prunedx, dim=-1)
               Var[i]=torch.var(prunedx, dim=-1)
        """
        eps=1e-20
        beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        bn3= ((self.gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+self.beta
        #bn3= (((xorig-Mean))/(Var+eps)      )
        
        return bn3

data_path='./imagenet/imagenet_tmp/raw_data/train'
dataset_train, dataset_test = ImagenetLoader.get_imagenet_datasets(data_path)

print(f"Number of train samplest {dataset_train.__len__()}")
print(f"Number of samples in test split {dataset_test.__len__()}")

BATCH_SIZE = 16

data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True, num_workers=12)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


   

def make_layers(cfg, batch_norm='None'):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]                
            elif v == 'Mc':
                layers += [ContinuousPool(in_channels)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm=='None':
                    layers += [conv2d, nn.ReLU(inplace=True)]                    
                elif batch_norm=="Normal":
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif batch_norm=="Double":
                    layers += [conv2d, BN2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

#continuous pool was only added to the later layers
cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'Mc', 512, 512, 512, 'Mc', 512, 512, 512, 'Mc']
batch_norm='Normal'
clf = VGG(make_layers(cfg_vgg16, batch_norm=batch_norm))
clf.cuda()        
#PATH="vgg16_4_"+batch_norm+".pt"
#clf.load_state_dict(torch.load(PATH),strict=False)

#clf.load_state_dict(torch.load('vgg16_5_Normal.pt'),strict=False)
#clf.load_state_dict(torch.load('vgg16_5_Normal.pt'))

criterion = nn.CrossEntropyLoss()

for param in clf.parameters():
        param.requires_grad = True
opt = optim.Adam(clf.parameters(), lr=0.001)

train_loss_history = []
train_acc1_history = []
train_acc5_history = []                  


def train(epoch):
    clf.train()
    for batch_id, batch in enumerate(data_loader_train):
        data=batch[0].cuda()
        label=batch[1].cuda()
        opt.zero_grad()
        preds =clf(data)
        loss = criterion(preds, label)
        loss.backward()
        opt.step()
        _, predind1 = preds.data.max(1)
        _, predind5 = torch.topk(preds.data,k=5, dim=1)

        acc1 = predind1.eq(label.data).float().mean().cpu() 
        
        label5=torch.unsqueeze(label.data,1)
        label5=label5.data.expand_as(predind5)
        correct5,_= predind5.eq(label5).max(1)

        acc5 = correct5.float().mean().cpu() 
        
        if batch_id % 10== 0:
            print("Epoch: "+str(epoch)+" Batch: "+str(batch_id)+" Train Loss: "+str(loss.item())+" Acc1: "+str(acc1.item())+" Acc5: "+str(acc5.item()))
            train_loss_history.append(loss.item())
            train_acc1_history.append(acc1.item())
            train_acc5_history.append(acc5.item())
            
for epoch in range(0, 10):
        print("Epoch %d" % epoch)
        train(epoch)
        torch.save(clf.state_dict(),"vgg16_"+str(epoch)+ "_Cont"+str(batch_norm)+ ".pt")


np.save("Contcold"+str(batch_norm)+"_train_loss.npy",np.array(train_loss_history))
np.save("Contcold"+str(batch_norm)+"_train_acc1.npy",np.array(train_acc1_history))
np.save("Contcold"+str(batch_norm)+"_train_acc5.npy",np.array(train_acc5_history))

