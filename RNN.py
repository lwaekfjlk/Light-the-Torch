import pandas as pd
from pandas_datareader import wb
import torch
import torch.nn
import torch.optim
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self,input_size,hidden_size):
    super(Net, self).__init__()
    self.rnn = torch.nn.RNN(input_size,hidden_size)
    self.fc = torch.nn.Linear(hidden_size,1)

  def forward(self,x):
    x = x[:,:,None]
    x, _ = self.rnn(x)
    x = self.fc(x)
    x = x[:,:,0]
    return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net(input_size = 1, hidden_size = 5)
net.to(device)
print(net)

# read the input data
countries = ['BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'IL', 'JP', 'SA', 'GB', 'US',]
dat = wb.download(indicator='NY.GDP.PCAP.KD',country=countries, start=1970, end=2016)
df = dat.unstack().T
df.index = df.index.droplevel(0).astype(int)
df_norm = df / df.loc[2000]
years = df.index
train_seq_len = sum((years >= 1971) & (years <= 2000))
test_seq_len = sum(years > 2000)

inputs = torch.tensor(df_norm.iloc[:-1].values,dtype=torch.float32).to(device)
labels = torch.tensor(df_norm.iloc[1:].values,dtype=torch.float32).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

iteration_num = 10000
plt_x1 = []
plt_x2 = []
plt_y1 = []
plt_y2 = []
for step in range(iteration_num):
  preds = net(inputs)
  train_preds = preds[:train_seq_len]
  test_preds = preds[train_seq_len:]
  train_labels = labels[:train_seq_len]
  test_labels = labels[train_seq_len:]

  train_loss = criterion(train_preds,train_labels)
  test_loss = criterion(test_preds,test_labels)

  optimizer.zero_grad()
  train_loss.backward()
  optimizer.step()

  plt_x1.append(step)
  plt_x2.append(step)
  plt_y1.append(train_loss)
  plt_y2.append(test_loss)
  if step % 1000 == 0:
    print("step = {} train_loss = {} test_loss = {}".format(step,train_loss,test_loss))

# show training-validation graph
plt.title("train-validation curve")
plt.plot(plt_x1, plt_y1,color="red",label="train")
plt.plot(plt_x2, plt_y2,color="blue",label="test")
plt.savefig("RNN.png")




