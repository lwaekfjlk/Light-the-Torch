import torch
import torch.nn as nn
import torch.optim
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):

  def __init__(self,input_size,hidden_size,output_size=1,cell_size=2):
    super(LSTM,self).__init__()
    self.hidden_size = hidden_size
    self.cell_size  = cell_size
    # just a matrix x*W.T + b
    # W shape is (cell_size, input_size+hidden_size)
    self.gate = nn.Linear(input_size + hidden_size, cell_size)
    # just a matrix x*W.T + b
    # W shape is (output_size, hidden_size)
    self.output = nn.Linear(cell_size, output_size)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.softmax = nn.LogSoftmax()

  def forward(self,input,hidden,cell):
    # input/hidden shoulde be shape (example_number, input/hidden_size)
    # here input/hidden_size means the length of input sequence
    # W_i*x_i+W_h*x_h + b = (input_matrix  hidden_matrix) * W.T + b

    combined = torch.cat((input,hidden),1)

    # f_gate = sigmoid(W_i*x_i+W_h*x_h + b)
    f_gate = self.sigmoid(self.gate(combined))
    # i_gate = sigmoid(W_i*x_i+W_h*x_h + b)
    i_gate = self.sigmoid(self.gate(combined))
    # o_gate = sigmoid(W_i*x_i+W_h*x_h + b)
    o_gate = self.sigmoid(self.gate(combined))
    # cell_sub = tanh(W_i*x_i+W_h*x_h + b)
    cell_sub = self.tanh(self.gate(combined))
    # cell_(t) = f_gate (dot) cell_(t-1) + i_gate (dot) cell_sub
    cell = torch.add(torch.mul(cell,f_gate),torch.mul(cell_sub,i_gate))
    # hidden = o_gate (dot) tanh(cell)
    hidden = torch.mul(self.tanh(cell), o_gate)
    # do one fc layer to get output
    output = self.sigmoid(self.output(hidden))
    return output,hidden,cell

  def initHidden(self,dim_num):
    # initialize one (eg_number, hidden_size) tensor
    return Variable(torch.zeros(dim_num,self.hidden_size))

  def initCell(self,dim_num):
    # initialize one (eg_number, cell_size) tensor
    return Variable(torch.zeros(dim_num,self.cell_size))

#=========================================================================


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x).squeeze(2),np.array(y)

# get init data
data  = pd.read_csv('./shampoo.csv')
data  = data.iloc[:,1:2].values
# do regulization to data
sc = MinMaxScaler()
training_data = sc.fit_transform(data)

# preprocess data
seq_length = 4 # means use 4 features in the data as output seq_length = input_size of LSTM
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# init the LSTM
input_size = seq_length
hidden_size = 2*seq_length
# compare to standard nn.LSTM
# input and output size is different
# the standard API ignore the seq_length = features of data
# their data are in shape [:,:,:] instead of [:,:]  in our case
lstm = LSTM(seq_length,2*seq_length)

# train
epoch_num = 10000
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(),0.01)

for epoch in range(epoch_num):
  # need to initialize the hidden matrix correctly, get (dim_num, feature_size) matrix
  outputs,_,_ = lstm(trainX,lstm.initHidden(trainX.shape[0]),lstm.initCell(trainX.shape[0]))
  optimizer.zero_grad()
  loss = criterion(outputs,trainY)
  loss.backward()
  optimizer.step()
  if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# eval
lstm.eval()
train_predict,_,_ = lstm(dataX,lstm.initHidden(dataX.shape[0]),lstm.initCell(dataX.shape[0]))

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()
# cancel regulization to data
data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

