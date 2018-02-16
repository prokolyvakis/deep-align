import theano
import numpy as np
from theano import config
from random import shuffle
from time import time
from tree import tree
from evaluate import evaluate
import pickle
import sys

def xrange(x):
    return iter(range(x))

def noneCheck(x, y):
    if x is None and y is None:
        return True
    return False

def checkIfQuarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

def saveParams(model, fname):
    f = file(fname, 'wb')
    pickle.dump(model.all_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def loadParams(model, fname):
    with open(fname, "rb") as input_file:
        model.all_params = pickle.load(input_file)
    return model.all_params

def prepare_data(list_of_seqs, max_len=None):

    if not list_of_seqs:
        return None, None
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = int(np.max(lengths))
    #print(type(maxlen))
    #print(type(max_len))
    if max_len: 
        maxlen = max(maxlen, max_len)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype=config.floatX)
    return x, x_mask

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def pickleLoad(file_dir):
    with open(file_dir, "rb") as input_file:
        data = pickle.load(input_file)
    return data

def getDataset(f,words,trans=lambda x: 1 if x == 'E' else 0,shuffle_num=1024):
    data = open(f,'r')
    lines = data.readlines()
    for _ in range(shuffle_num):
        shuffle(lines)
    
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                e = (tree(i[0], words), tree(i[1], words), trans(i[2]))
                examples.append(e)
                e_ = (tree(i[1], words), tree(i[0], words), trans(i[2]))
                examples.append(e_)
            else:
                print(i)
    return examples

def createSet(examples, words):
    res = set()
    result = []
    
    for x in examples:
        x[0].populate_embeddings(words)
        phrase = x[0].phrase
        if not phrase in res:
            res.add(phrase)
            result.append(x[0])
    
    return result

def getAntRepresentations(model, antonyms):
    g = []

    for ant in antonyms:
        g.append(ant.embeddings)
    gx, gmask = prepare_data(g)
    embg = model.feedforward_function(gx, gmask)

    for idx, ant in enumerate(antonyms):
        ant.representation = embg[idx, :]
    return

def getAntonyms(textfile,words):
    with open(textfile,'r') as f:
        lines = f.readlines()

    antonyms = []
    for line in lines:
        if len(line) > 0:
            e = tree(line.rstrip(), words)
            e.populate_embeddings(words)
            antonyms.append( e )
    
    return antonyms

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def getpairs(model, batch, params):
    g1 = []; p1 = []
    g2 = []; p2 = []

    for i in batch:
        if i[2] == 0:
            g1.append(i[0].embeddings)
            g2.append(i[1].embeddings)
        else:
            p1.append(i[0].embeddings)
            p2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)
    p1x, p1mask = prepare_data(p1)
    p2x, p2mask = prepare_data(p2)

    return (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask)