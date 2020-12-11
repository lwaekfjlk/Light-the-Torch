from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# GAN is sensitive to parameter initialization
# we need to choose appropriate parameters to do initialization
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


dataset = CIFAR10(root = "./data",transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

for batch_idx, data in enumerate(dataloader):
  real_images = data[0].to(device)
  batch_size = real_images.size(0)
  if batch_idx % 300 == 0:
        print("current handled batch : {}".format(batch_idx))
        path = './data/CIFAR10_shuffled_batch{:03d}.png'.format(batch_idx)
        save_image(real_images, path, normalize=True)

#=========================
# generate network
#=========================

# batch_size of input data
in_channel = 64
# output channel
out_channel = 3
# size of picture
generate_feature = 64

# since the generate net want to transform the input of (64,1,1) data to (3, 32,32) data
# we use the stride = 2 and transpose2D conv to enlarge the 1 to 32
# we contro the input and output channel of conv layer to control the channel of the net
layers = []
layers.append(nn.ConvTranspose2d(in_channel,4*generate_feature,kernel_size=4,bias=False))
layers.append(nn.BatchNorm2d(4*generate_feature))
layers.append(nn.ReLU())
layers.append(nn.ConvTranspose2d(4*generate_feature,2*generate_feature,kernel_size=4,stride=2,padding=1,bias=False))
layers.append(nn.BatchNorm2d(2*generate_feature))
layers.append(nn.ReLU())
layers.append(nn.ConvTranspose2d(2*generate_feature,generate_feature,kernel_size=4,stride=2,padding=1,bias=False))
layers.append(nn.BatchNorm2d(generate_feature))
layers.append(nn.ReLU())
layers.append(nn.ConvTranspose2d(generate_feature,out_channel,kernel_size=4,stride=2,padding=1))
layers.append(nn.Tanh())
# build network
generate_net = nn.Sequential(*layers)
# init parameters
generate_net.apply(weights_init)
# put into GPU
generate_net.to(device)

#=========================
# discriminate network
#=========================
in_channel = 3
out_channel = 1
discriminate_feature = 64
layers = []
layers.append(nn.Conv2d(in_channel,discriminate_feature,kernel_size=4,stride=2,padding=1))
layers.append(nn.LeakyReLU(0.2))
layers.append(nn.Conv2d(discriminate_feature,2*discriminate_feature,kernel_size=4,stride=2,padding=1))
layers.append(nn.BatchNorm2d(2*discriminate_feature))
layers.append(nn.LeakyReLU(0.2))
layers.append(nn.Conv2d(2*discriminate_feature,4*discriminate_feature,kernel_size=4,stride=2,padding=1))
layers.append(nn.BatchNorm2d(4*discriminate_feature))
layers.append(nn.LeakyReLU(0.2))
layers.append(nn.Conv2d(4*discriminate_feature,1,kernel_size=4))
# build network
discriminate_net = nn.Sequential(*layers)
# init parameters
discriminate_net.apply(weights_init)
# put into GPU
discriminate_net.to(device)


criterion = nn.BCEWithLogitsLoss()
generate_optimizer = torch.optim.Adam(generate_net.parameters(),lr=0.0002, betas=(0.5, 0.999))
discriminate_optimizer = torch.optim.Adam(discriminate_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

epoch_num = 10

for epoch in range(epoch_num):
  for batch_idx, data in enumerate(dataloader):
    batch_size = data[0].size(0)
    pic_size = 64

    # load some real images
    real_images = data[0].to(device)
    # feed real images into discriminate net
    preds = discriminate_net(real_images).view(-1)
    labels = torch.ones(batch_size).to(device)
    dloss_real = criterion(preds, labels)
    dmean_real = preds.sigmoid().mean()

    # generate some fake images
    noises = torch.randn(batch_size,pic_size,1,1).to(device)
    fake_images = generate_net(noises)
    # grad info not go backward to generate_net
    # accelerate the training speed
    fake = fake_images.detach()
    # feed fake images into discriminate net
    preds = discriminate_net(fake).view(-1) # torch uses view(-1) while numpy uses reshape(1,-1)
    labels = torch.zeros(batch_size).to(device)
    dloss_fake = criterion(preds, labels)
    dmean_fake = preds.sigmoid().mean()

    # train the discriminate-net
    dloss = dloss_real + dloss_fake
    discriminate_optimizer.zero_grad()
    dloss.backward()
    discriminate_optimizer.step()

    # train the generate-nte
    preds = discriminate_net(fake_images).view(-1)
    labels = torch.ones(batch_size).to(device)
    gloss = criterion(preds, labels)
    gmeans_fake = preds.sigmoid().mean()
    generate_optimizer.zero_grad()
    gloss.backward()
    generate_optimizer.step()

    print("epoch = {} batch_percent = {}, dloss = {} gloss = {}".format(epoch,batch_idx/len(dataloader),dloss,gloss))
    fixed_noises = torch.randn(batch_size, pic_size, 1, 1).to(device)
    if (batch_idx % 300 == 0):
      fake = generate_net(fixed_noises)
      print("Time to save something awesome!")
      save_image(fake,'./data/epoch_{:02d}_batch_{:02d}.png'.format(epoch,batch_idx))


