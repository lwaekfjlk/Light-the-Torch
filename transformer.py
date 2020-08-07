import numpy as np 
import torch
import torch.nn as nn
import torch.optim 
import torch.nn.functional as F
from torch.autograd import Variable
import copy,math,time


def clones(module, N):
  # return a list of copied  module
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
  # size of vocab
  d_k = query.size(-1)
  # (Q*K.T)/ sqrt(d_k)
  scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
  # fill zero part of the mask to be -INF
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

  # softmax((Q*K.T)/ sqrt(d_k))
  p_attn = F.softmax(scores, dim = -1)

  if dropout is not None:
    p_attn = dropout(p_attn)

  # return attention and softmax((Q*K.T)/ sqrt(d_k))
  return torch.matmul(p_attn, value), p_attn

#==================
# module 1
#==================
class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    self.d_k = d_model // h
    self.h = h
    # Linear[0] = W_V
    # Linear[1] = W_Q
    # Linear[2] = W_K
    # Linear[3] = W_final linear matrix
    self.linears = clones(nn.Linear(d_model,d_model),4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    # number of vocab = size of one batch
    nbatches = query.size(0)

    # dimensions for this part
    # d_q = d_k = d_v = d_model // h = 64

    # 1. Linear Projection
    # Q,K,V  =  nbatches * n_model
    # W = n_model * n_model
    # Q * W = nbatches * n_model
    # transform Q * W to nbatches * (h * 1 * d_k)
    query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l,x in zip(self.linears, (query, key, value))]

    # 2. Do Attention  attn(QW,KW,VW)
    # x = softmax(QW*KW.T / sqrt(d_k)) * VW
    # x.shape = nbatches * (h * 1 * d_k)
    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

    # 3. Concat
    # x.shape = nbatches * (1 * h * d_k)
    # x.shape = nbatches * 1 * (h * d_k) = nbatchhes * 1 * n_model
    x = x.transpose(1,2).contiguous().view(nbatches,-1, self.h * self.d_k)

    # 4. Linear
    return self.linears[-1](x)

#==================
# module 2
#==================
class PositionwiseFeedForward(nn.Module):
  def __init__(self,d_model,d_ff,dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    # layer1 --- dropout_layer --- layer2
    self.w_1 = nn.Linear(d_model,d_ff)
    self.w_2 = nn.Linear(d_ff,d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.w_1(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.w_2(x)
    return x

#==================
# submodule 4
#==================
class LayerNorm(nn.Module):
  def __init__(self,features,eps=1e-6):
    super(LayerNorm,self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self,x):
    # x = (x - mean) / (std + eps)
    # just do an norm
    # input of LayerNorm should be a resdual part
    # x + submodule(x) ====> input of LayerNorm
    mean = x.mean(-1,keepdim=True)
    std = x.std(-1,keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#==================
# module 4
#==================
class SublayerConnection(nn.Module):
  def __init__(self,size,dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x,sublayer):
    # create the input for LayerNorm
    # x + (sublayer(norm(x))) wrap the LayerNorm 
    # Become the Add & norm part of the model
    return x + self.dropout(sublayer(self.norm(x)))

#==================
# Encoder Layer Part
#==================
class EncoderLayer(nn.Module):
  def __init__(self,size,self_attn,feed_forward,dropout):
    super(EncoderLayer,self).__init__()
    # attention module
    self.self_attn = self_attn
    # feed_forward module
    self.feed_forward = feed_forward
    # two sublayers needs two sublayerConnection to perform residual operations
    self.sublayer = clones(SublayerConnection(size,dropout),2)
    self.size  = size

  def forward(self,x,mask):
    #  x --- attention --- add&norm --- feed_forward --- add&norm
    x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
    x = self.sublayer[1](x, self.feed_forward)
    return x

#==================
# Encoder Whole Part
#==================
class Encoder(nn.Module):
  def __init__(self,layer,N):
    super(Encoder,self).__init__()
    self.layers = clones(layer,N)
    self.norm = LayerNorm(layer.size)

  def forward(self,x,mask):
    # create N layers for encoderlayer
    # x --- norm(Encoderlayer(x))
    for layer in self.layers:
      x = layer(x,mask)
    return self.norm(x)

#==================
# Decoder Layer Part
#==================
class DecoderLayer(nn.Module):
  def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
    super(DecoderLayer,self).__init__()
    # attention module
    self.self_attn = self_attn
    self.src_attn = src_attn
    # feed_forward module
    self.feed_forward = feed_forward
    # three sublayers needs three sublayerConnection to perform residual operations
    self.sublayer = clones(SublayerConnection(size,dropout),3)
    self.size  = size

  def forward(self,x,memory,src_mask,tgt_mask):
    #  x --- attention --- add&norm --- (input memory)attention --- add&norm --- feed_forward --- add&norm
    m = memory
    x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,tgt_mask))
    x = self.sublayer[1](x, lambda x:self.src_attn(x,m,m,src_mask))
    x = self.sublayer[2](x, self.feed_forward)
    return x

#==================
# Decoder Whole Part
#==================
class Decoder(nn.Module):
  def __init__(self,layer,N):
    super(Decoder,self).__init__()
    self.layers = clones(layer,N)
    self.norm = LayerNorm(layer.size)

  def forward(self,x,memory,src_mask,tgt_mask):
    # create N layers for encoderlayer
    # x --- norm(Decoderlayer(x))
    for layer in self.layers:
      x = layer(x,memory,src_mask,tgt_mask)
    return self.norm(x)

#==================
# Generator Whole Part
#==================
class Generator(nn.Module):
  def __init__(self,d_model,vocab):
    super(Generator,self).__init__()
    self.proj = nn.Linear(d_model,vocab)
  def forward(self,x):
    # handle the last part of transformer
    # project dim=d_model result into dim=vocab result
    return F.log_softmax(self.proj(x),dim=-1)

#==================
# Complete Model
#==================
class EncoderDecoder(nn.Module):

  def __init__(self,encoder,decoder,src_embed, tgt_embed, generator):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator

  def forward(self, src, tgt, src_mask, tgt_mask):
    return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

  def encode(self,src,src_mask):
    return self.encoder(self.src_embed(src),src_mask)

  # this memory comes from the encoder part of transformer
  def decode(self,memory,src_mask,tgt,tgt_mask):
    return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

#==================
# Embedding
#==================
class Embeddings(nn.Module):
  def __init__(self,d_model,vocab):
    super(Embeddings, self).__init__()
    # input the size of dict, output required dim=d_model embedding vector
    self.lut = nn.Embedding(vocab,d_model)
    self.d_model = d_model
  def forward(self,x):
    # use sqrt(d_model) to regularize the embedding
    return self.lut(x) * math.sqrt(self.d_model)

#==================
# Position Embedding
#==================
class PositionalEncoding(nn.Module):
  def __init__(self,d_model,dropout,max_len=5000):
    super(PositionalEncoding,self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len,d_model)
    position = torch.arange(0,max_len).unsqueeze(1)

    # div_term = e^(2i/ d_model * (-log(10000))) = 10000^(-2*i/d_model) = 1/(10000^(2i/d_model))
    div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0))/d_model)

    # pe(2i) = sin(pos / 10000^{2*i/d_model})
    pe[:, 0::2] = torch.sin(position * div_term)
    # pe(2i+1) = cos(pos / 10000^{2*i/d_model})
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe',pe)

  def forward(self,x):
    x += Variable(self.pe[:,:x.size(1)], requires_grad=False)
    return self.dropout(x)

#==================
# Whole Model
#==================
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
          Encoder(EncoderLayer(d_model, c(attn),c(ff),dropout),N),
          Decoder(DecoderLayer(d_model, c(attn),c(attn),c(ff),dropout),N),
          nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
          nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
          Generator(d_model, tgt_vocab)
          )

  # init parameters
  # very important for the results of the model
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model

#==============================================

#==================
# Better Optimizer
#==================
class NoamOpt:
  # change the learning rate of the optimizer Adam
  # other things keep the same
  def __init__(self, model_size, factor, warmup, optimizer):
    self.optimizer = optimizer
    self._step = 0
    self.warmup = warmup
    self.factor = factor
    self.model_size  = model_size
    self._rate = 0

  def step(self):
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      # change learning rate
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step=None):
    if step is None:
      step = self._step
    return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#==============================================

class Batch:
  def __init__(self,src,tgt=None,pad=0):
    self.src = src
    # create the mask for src
    self.src_mask = (src != pad).unsqueeze(-2)

    if tgt is not None:
      # tgt is the target sequence, the input of decoder

      # prepare the data and the labels for RNN-like
      self.tgt = tgt[:,:-1]
      self.tgt_y =  tgt[:,1:]

      # padding mask
      self.tgt_mask = self.make_std_mask(self.tgt, pad)
      self.ntokens = (self.tgt_y != pad).data.sum()

  @staticmethod 
  def make_std_mask(tgt,pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def subsequent_mask(size):
    # provide subsequent mask in order to not let decoder see the words behind him
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

#==================
# generate data
#==================
def data_gen(V, batch, nbatches):
  # each i create one batch
  for i in range(nbatches):
    data = torch.from_numpy(np.random.randint(1,V,size=(batch,10)))
    # to make sure the beginning of this sequence is definitely 1
    data[:,0] = 1

    # the source array and target array is the same
    src = Variable(data, requires_grad=False)
    tgt = Variable(data, requires_grad=False)
    yield Batch(src, tgt, 0)

#==============================================

#==================
# calculate loss
#==================
class SimpleLossCompute:
  def __init__(self, generator, criterion, opt=None):
    self.generator = generator
    self.criterion = criterion
    self.opt = opt
  def __call__(self,x,y,norm):
    x = self.generator(x)
    preds = x.contiguous().view(-1,x.size(-1))
    target = y.contiguous().view(-1)
    labels = torch.zeros(preds.shape)
    labels.scatter_(1,target.data.unsqueeze(1),1)
    loss = self.criterion(preds.float(),labels.float()) / norm

    if self.opt is not None:
      self.opt.optimizer.zero_grad()
    loss.backward()
    if self.opt is not None:
      self.opt.step()

    return loss.data * norm

#==============================================

#==================
# run an epoch
#==================
def run_epoch(data_iter, model, loss_compute):
  total_tokens = 0
  total_loss = 0
  print(data_iter.shape)
  for i,batch in enumerate(data_iter):
    out = model.forward(batch.src, batch.tgt,batch.src_mask,batch.tgt_mask)

    # out is logits before Generator 
    loss = loss_compute(out, batch.tgt_y, batch.ntokens)

    total_loss += loss
    total_tokens += batch.ntokens
    if (i % 50 == 0):
      print("step = {} , loss per token = {}".format(i,loss/batch.ntokens))

  return total_loss / total_tokens

#==============================================

def greedy_decode(model, src, src_mask, max_len, start_symbol):
  memory = model.encode(src, src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
  for i in range(max_len-1):
      out = model.decode(memory, src_mask, 
                         Variable(ys), 
                         Variable(subsequent_mask(ys.size(1))
                                  .type_as(src.data)))
      prob = model.generator(out[:, -1])
      _, next_word = torch.max(prob, dim = 1)
      next_word = next_word.data[0]
      ys = torch.cat([ys, 
                      torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
  return ys

#==============================================

def main():
  V = 11
  criterion = nn.KLDivLoss(size_average=False)
  model = make_model(V, V, N=2)
  model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
  
  model.eval()
  src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
  src_mask = Variable(torch.ones(1, 1, 10) )
  print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


if __name__ == '__main__':
  main()

