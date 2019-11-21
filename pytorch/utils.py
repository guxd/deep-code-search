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
import nltk
try: nltk.word_tokenize("hello world")
except LookupError: nltk.download('punkt')
    
def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''      
    def convert_sent(sent, vocab, maxlen):
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        tokens = nltk.word_tokenize(sent.strip())
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len): idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len
    if type(sentence) is list:
        inds, lens = None, None
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            idxes, idx_len = np.expand_dims(idxes, 0), np.array([idx_len])
            inds = idxes if inds is None else np.concatenate((inds, idxes))
            lens = idx_len if lens is None else np.concatenate((lens, idx_len))
        return inds, lens
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]

########################################################################
