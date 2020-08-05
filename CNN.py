import torch
import torch.utils.data
import torch.nn
import torch.optim
import torchvision.datasets
import torchvision.transforms

class Net(torch.nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv0 = torch.nn.Conv2d(1,64,kernel_size = 3,padding = 1)
    self.relu1 = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv2d(64,128,kernel_size = 3, padding = 1)
    self.relu3 = torch.nn.ReLU()
    self.pool4 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.fc5 = torch.nn.Linear(128 * 14 * 14, 1024)
    self.relu6 = torch.nn.ReLU()
    self.drop7 = torch.nn.Dropout(p=0.5)
    self.fc8 = torch.nn.Linear(1024,10)
  def forward(self,x):
    x = self.conv0(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu3(x)
    x = self.pool4(x)
    x = x.view(-1,128*14*14)
    x = self.fc5(x)
    x = self.relu6(x)
    x = self.drop7(x)
    x = self.fc8(x)
    return x


train_dataset = torchvision.datasets.MNIST(root='./data/mnist',train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/mnist',train=False, transform=torchvision.transforms.ToTensor(),download=True)

# pack the dataset into batch according to the batch size
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)


# get ready for training
net = Net()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

epoch_size = 5
for epoch in range(epoch_size):
  for idx, data in enumerate(train_loader):
    image_data = data[0].to(device)
    labels = data[1].to(device)
    preds = net(image_data)
    loss = criterion(preds,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if idx % 100 == 0:
      print("epoch = {} iteration = {} loss = {}".format(epoch,idx,loss))

total = 0
correct = 0
for data in test_loader:
  image_data = data[0].to(device)
  labels = data[1].to(device)
  preds = net(image_data)
  # select the largest one as our predition result
  predicted = torch.argmax(preds,1)
  # count samples handled
  total += labels.size(0)
  # count correct ones
  correct += (predicted == labels).sum().item()

acc = correct / total
print("acc is {}".format(acc))
