import numpy as np
import time
import math
import torch
from torch.nn import functional as F

PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]

def cos_np(data1,data2):
    """numpy implementation of cosine similarity for matrix"""
    dotted = np.dot(data1,np.transpose(data2))
    norm1 = np.linalg.norm(data1,axis=1)
    norm2 = np.linalg.norm(data2,axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data/np.linalg.norm(data,axis=1).reshape((data.shape[0], 1))
    return normalized_data

def dot_np(data1,data2):
    """cosine similarity for normalized vectors"""
    return np.dot(data1,np.transpose(data2))

#######################################################################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d'% (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asMinutes(s), asMinutes(rs))

#######################################################################

def sent2indexes(sentence, vocab, max_len=None):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''
    def convert_sent(sent, vocab):
        return np.array([vocab.get(word, UNK_ID) for word in sent.split()])
    if type(sentence) is list:
        indexes=[convert_sent(sent, vocab) for sent in sentence]
        sent_lens = [len(idxes) for idxes in indexes]
        if max_len is None:
            max_len = max(sent_lens)
        inds = np.zeros((len(sentence), max_len), dtype=np.int)
        for i, idxes in enumerate(indexes):
            inds[i,:len(idxes)]=indexes[i][:max_len]
        return inds
    else:
        return convert_sent(sentence, vocab)

########################################################################
