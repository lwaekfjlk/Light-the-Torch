from time import time
import copy
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn.functional as F
from torch import nn
import numpy as np


def train(model, criterion, optimizer, trainLoader, valLoader, num_epochs):
    """
    model:网络模型；criterion：损失函数；optimizer：优化方法；
    traindataloader:训练数据集，valdataloader:验证数据集
    num_epochs:训练的轮数
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all, train_acc_all, train_IoU_all, val_IoU_all = [], [], [], []
    val_loss_all, val_acc_all = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Train on", device)
    model.to(device)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs), "-" * 10, sep="\n")
        train_loss, val_loss, train_IoU, val_IoU = 0.0, 0.0, 0.0, 0.0
        train_num, val_num = 0, 0
        # 每个epoch包括训练和验证阶段
        model.train()  ## 设置模型为训练模式
        for step, (b_x, b_y) in tqdm(enumerate(trainLoader)):
            optimizer.zero_grad()
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)
            train_IoU += sum(calIoU(pre_lab, b_y))
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_IoU_all.append(train_IoU / train_num)
        print(
            "EPOCH{}: Train Loss: {:.4f}, Train IoU: {:.3f}".format(
                epoch, train_loss_all[-1], train_IoU_all[-1]
            )
        )

        ## 计算一个epoch的训练后在验证集上的损失
        model.eval()  ## 设置模型为训练模式评估模式
        for step, (b_x, b_y) in tqdm(enumerate(valLoader)):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
            val_IoU += sum(calIoU(pre_lab, b_y))
        ## 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        val_IoU_all.append(val_IoU / val_num)
        print(
            "EPOCH{}: Val Loss: {:.4f}, Val IoU".format(
                epoch, val_loss_all[-1], val_IoU_all[-1]
            )
        )
        ## 保存最好的网络参数
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
        }
    )
    ## 输出最好的模型
    model.load_state_dict(best_model_wts)
    return model, train_process


def calIoU(prediction, label):
    assert prediction.shape == label.shape
    prediction, label = nn.Flatten()(prediction), nn.Flatten()(label)
    res = []
    for i in range(label.shape[0]):
        TP, FP, TN, FN = 0, 0, 0, 0
        P, L = np.array(prediction[i].cpu()), np.array(label[i].cpu())
        for j in range(len(P)):
            datatype = P[j] * 2 + L[j]
            if datatype == 0:
                TN += 1
            elif datatype == 1:
                FN += 1
            elif datatype == 2:
                FP += 1
            elif datatype == 3:
                TP += 1
            else:
                raise ValueError('the value of two tensors should be 0 or 1')
    res.append(TP / (TP + FP + FN))
    return res
