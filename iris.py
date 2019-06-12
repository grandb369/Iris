import sys
import logging
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.datasets.folder import IMG_EXTENSIONS
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm,tqdm_notebook

logging.basicConfig(stream=sys.stdout)
logger=logging.getLogger("log")
logger.setLevel(logging.INFO)
lg=logging.FileHandler('log.txt')
lg.setLevel(logging.INFO)
logger.addHandler(lg)



device=torch.cuda.is_available()
fold='leftright_LG2200/'
batch_size=64
lr=0.025
lr_min=0.001
channels=64
momentum=0.9
gamma=0.97
epochs=500
classes=90
weight_decay=3e-4
portion=0.7
#IMG_EXTENSIONS.append('tiff')

class Net(nn.Module):
    def __init__(self,C,classes):
        super(Net,self).__init__()
        self.s1=nn.Sequential(nn.Conv2d(3,C//2,kernel_size=3,stride=2,padding=1,bias=False),
                             nn.BatchNorm2d(C//2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(C//2,C,3,stride=2,padding=1,bias=False),
                             nn.BatchNorm2d(C),)
        self.s2=nn.Sequential(nn.ReLU(inplace=True),
                             nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1,bias=False),
                             nn.BatchNorm2d(C),)
        self.sep=nn.Sequential(nn.ReLU(inplace=False),
                              nn.Conv2d(C,C,kernel_size=3,stride=2,padding=1,bias=False),
                              nn.Conv2d(C,C,kernel_size=1,padding=0,bias=False),
                              nn.BatchNorm2d(C),
                              nn.ReLU(inplace=False),
                              nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1,bias=False),
                              nn.Conv2d(C,C,kernel_size=1,padding=0,bias=False),
                              nn.BatchNorm2d(C),)
        self.ReConBn=nn.Sequential(nn.ReLU(inplace=False),
                                  nn.Conv2d(C,C,kernel_size=1,stride=1,padding=0,bias=False),
                                  nn.BatchNorm2d(C))
        self.maxpool=nn.MaxPool2d(2)
        self.pooling=nn.AvgPool2d(7)
        self.out=nn.Linear(C,classes)
    
    def forward(self,x):
        c1=self.s1(x)
        c2=self.s2(c1)
        #print(c1.shape,c2.shape)
        node0_c1=self.sep(c1)
        node0_c2=self.sep(c2)
        #print(node0_c1.shape,node0_c2.shape)
        node0=node0_c1+node0_c2
        node1_c1=self.maxpool(c1)
        node1_c2=self.sep(c2)
        #print(node1_c1.shape,node1_c2.shape)
        node1=node1_c1+node1_c2
        node2_c1=self.maxpool(c1)
        node2_c2=self.sep(c2)
        #print(node2_c1.shape,node2_c2.shape)
        node2=node2_c1+node2_c2
        node3_1=node0
        node3_2=self.sep(c1)
        #print(node3_1.shape,node3_2.shape)
        node3=node3_1+node3_2
        #print(node0.shape,node1.shape,node2.shape,node3.shape)
        out_node=node0+node1+node2+node3
        out=self.pooling(out_node)
        #print(out.shape)
        out=self.maxpool(self.maxpool(out))
        #print(out.shape)

        out=out.view(out.size(0),-1)
        #print(out.shape)
        out=self.out(out)
        return out
        
model=Net(C=channels,classes=classes)
loss_fuc=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
if device:
    model.cuda()
    loss_fuc.cuda()
    
    
train_transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
valid_transform=transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
train_data=dset.ImageFolder(fold,train_transform)
num_train=len(train_data)
indices=list(range(num_train))
split=int(np.floor(portion*num_train))

train_queue=torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                       pin_memory=False,num_workers=0)
valid_queue=torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                                       pin_memory=False,num_workers=0)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,float(epochs),eta_min=lr_min)

def train(train_queue,model,loss_fuc,optimizer):
    pre=[]
    label=[]
    for step,(data,target)in enumerate(train_queue):
        model.train()
        label+=list(target.numpy())
        data=Variable(data)
        target=Variable(target)
        if device:
            data=data.cuda()
            target=target.cuda()
        optimizer.zero_grad()
        out=model(data)
        temp=torch.max(out.data,1)[1]
        pre=(temp==target).sum()
        loss=loss_fuc(out,target)
        loss.backward()
        optimizer.step()
    acc=float(pre)/len(label)
    if device:
        logger.info('Train accuracy: {} \tLoss: {}'.format(acc,float(loss.data.cpu().numpy())))
        return float(loss.data.cpu().numpy()),acc 
    else:
        logger.info('Train accuracy: {} \tLoss: {}'.format(acc,float(loss.data.numpy())))
        return float(loss.data.numpy()),acc 

def validation(valid_queue,model,loss_fuc,optimizer):
    model.eval()
    pre=[]
    label=[]
    with torch.no_grad():
        for step,(data,target)in enumerate(train_queue):
            model.train()
            label+=list(target.numpy())
            data=Variable(data)
            target=Variable(target)
            if device:
                data=data.cuda()
                target=target.cuda()
            out=model(data)
            temp=torch.max(out.data,1)[1]
            pre=(temp==target).sum()
            loss=loss_fuc(out,target)
    acc=float(pre)/len(label)
    if device:
        logger.info('Valid accuracy: {} \tLoss: {}'.format(acc,float(loss.data.cpu().numpy())))
        return float(loss.data.cup().numpy()),acc 
    else:
        logger.info('Valid accuracy: {} \tLoss: {}'.format(acc,float(loss.data.numpy())))
        return float(loss.data.numpy()),acc 


tra_loss=[]
val_loss=[]
tra_acc=[]
val_acc=[]
for epoch in range(epochs):
    logger.info("Epoch: {}".format(epoch))
    scheduler.step()
    train_loss,train_acc=train(train_queue,model,loss_fuc,optimizer)
    tra_acc.append(train_acc)
    tra_loss.append(train_loss)
    valid_loss,valid_acc=validation(valid_queue,model,loss_fuc,optimizer)
    val_loss.append(valid_loss)
    val_acc.append(valid_acc)