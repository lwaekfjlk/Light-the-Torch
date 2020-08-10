import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import  matplotlib.pyplot as  plt
import os, time, copy



# fit a series of  transformation of data by adding transforms.Compose
data_transforms = {
  'train':
  # data augmentation
  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize mean, std
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
  'val':
  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data from data file
data_dir = {x: './hymenoptera_data/'+ x for x in ['train','val']}
dataset = {x: datasets.ImageFolder(data_dir[x], transform=data_transforms[x]) for x in ['train','val']}
dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train','val']}
dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}
class_names = {x: dataset[x].classes for x in ['train', 'val']}

# resnet18 model
model = models.resnet18(pretrained=True)
# change the last fully connect layer in VGG model
# to fit the classification type of the model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,len(class_names['train']))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
# modify lr and decay lr 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# choose the best model paramters based on acc on validation data
best_model_checkpoint = copy.deepcopy(model.state_dict())
best_acc = 0.0

epoch_num = 25
for epoch in range(epoch_num):
    begin_time = time.time()
    for phase in ['train','val']:
        if phase == 'train':
          # do everthing
          model.train()
        else:
          # do nothing on nn.dropout and nn.BatchNorm
          # forward and backward propagation still work
          model.eval()

        running_loss = 0.0
        running_correct = 0.0

        for inputs,labels in (dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # select the max one in the output of the model to be the final result of classification
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)

            if (phase == 'train'):
                loss.backward()
                # update the model in one batch
                optimizer.step()

            running_loss += loss.item() * inputs.shape[]
            running_correct += torch.sum(preds == labels.data).double()

        # decay the lr in one training epoch instead of optimizer.step() in one batch
        if (phase == 'train'):
            exp_lr_scheduler.step()

        epoch_loss = running_loss / dataset_size[phase]
        epoch_acc = running_correct / dataset_size[phase]

        print("{} Epoch: {} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(phase, epoch, epoch_loss, epoch_acc,time.time() - begin_time))

        if phase == 'val' and epoch_acc > best_acc :
            best_acc = epoch_acc
            best_model_checkpoint = copy.deepcopy(model.state_dict())

model = model.load_state_dict(best_model_checkpoint)






