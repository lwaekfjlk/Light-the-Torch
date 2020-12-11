import torch 
import torch.nn as nn
import torch.optim

def himmelblau(x):
    return (x[:,0] ** 2 + x[:,1] - 11) ** 2 + (x[:,0] + x[:,1] ** 2 - 7) ** 2


# generate data from function
torch.manual_seed(seed=0) # 固定随机数种子,这样生成的数据是确定的
sample_num = 1000 # 生成样本数
features = torch.rand(sample_num, 2)  * 12 - 6 # 特征数据
noises = torch.randn(sample_num)
hims = himmelblau(features) * 0.01
labels = hims + noises # 标签数据


hidden_features = [6,2]
layers = [nn.Linear(2,hidden_features[0])]
layers.append(nn.Sigmoid())
layers.append(nn.Linear(hidden_features[0],hidden_features[1]))
layers.append(nn.Sigmoid())
layers.append(nn.Linear(hidden_features[1],1))

# use nn.Sequential to have the default parameters and forward function for the existing net
net = nn.Sequential(*layers)

print("FCNN : {}".format(net))

# to get the optimizer and criterion of the net
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()

train_num = 600
validation_num = 200
test_num = 200
iteration_num = 20000
net_input = features
for step in range(iteration_num):
  net_output = net(features)
  net_pred = net_output.reshape(net_output.shape[0],)

  loss_train = criterion(net_pred[:train_num],labels[:train_num])
  loss_validation = criterion(net_pred[train_num:-validation_num],labels[train_num:-validation_num])

  optimizer.zero_grad()
  loss_train.backward()
  optimizer.step()

  if step % 1000 == 0:
    print("step = {} Training Data MSE = {:g}, Validation Data MSE = {:g}".format(step,loss_train,loss_validation))

