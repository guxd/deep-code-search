import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class BOWEncoder(nn.Module):
    '''
    https://medium.com/data-from-the-trenches/how-deep-does-your-sentence-embedding-model-need-to-be-cdffa191cb53
    https://www.kdnuggets.com/2019/10/beyond-word-embedding-document-embedding.html
    https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d#bbe8
    '''
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(BOWEncoder, self).__init__()
        self.emb_size=emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #self.word_weights = get_word_weights(vocab_size) 
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        
    def forward(self, input, input_len=None): 
        batch_size, seq_len =input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training) # [batch_size x seq_len x emb_size]
        
        # try to use a weighting scheme to summarize bag of word embeddings: 
        # for example, a smooth inverse frequency weighting algorithm: https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py
        # word_weights = self.word_weights(input) # [batch_size x seq_len x 1]
        # embeded = word_weights*embedded 
        
        # max pooling word vectors
        maxpooling = nn.MaxPool1d(kernel_size = seq_len, stride=seq_len)
        output_pool = maxpooling(embedded.transpose(1,2)).squeeze(2) # [batch_size x emb_size]
        encoding = output_pool #torch.tanh(output_pool)        
        return encoding
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        for name, param in self.lstm.named_parameters(): # initialize the gate weights 
            # adopted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
            #if len(param.shape)>1:
            #    weight_init.orthogonal_(param.data) 
            #else:
            #    weight_init.normal_(param.data)                
            # adopted from fairseq
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, input_lens=None): 
        '''
        input_lens: [batch_size]
        '''
        batch_size, seq_len=inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.25, self.training)
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs)  
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True) # hids:[batch_size x seq_len x (n_dir*hid_sz)](biRNN)
            hids = F.dropout(hids, p=0.25, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
############commenting the following line significantly improves the performance, why? #####################################
      #  h_n1 = h_n.transpose(1, 0).contiguous() #[batch_size x n_dirs x hid_sz]
      #  encoding1 = h_n1.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        
        #https://www.jianshu.com/p/c5b8e02bedbe
        #maxpooling = nn.MaxPool1d(kernel_size=hids.size(1), stride=hids.size(1))
        #encoding2 = maxpooling(hids.transpose(1,2)).squeeze(2) # [batch_size x 2*hid_size]
        #encoding2 = torch.tanh(encoding2)

        encoding3 = torch.cat((h_n[0], h_n[1]), dim=1)
        return encoding3 #, encoding2, encoding3

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)

 
 
 
 