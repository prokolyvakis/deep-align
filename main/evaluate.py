from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import theano
from theano import config
import utils
from tree import lookupIDX
import keras
import time

def getSeq(p,words):
    p = p.split()
    X = []
    for i in p:
        X.append(lookupIDX(words,i))
    return X

def getSeqs(p1,p2,words):
    return getSeq(p1,words), getSeq(p2,words)

def sen2Embgs(model,p,words):
    w2codes = getSeq(p,words)
    codes, _ = utils.prepare_data([w2codes])
    embgs = model.word2embeddings(codes)

    return dict(zip(p.split(), embgs[0]))

def predictSimilarity(model,p1,p2,words):
    X1, X2 = getSeqs(p1,p2,words)
    x1,m1 = utils.prepare_data([X1])
    x2,m2 = utils.prepare_data([X2])
    maxlen = m2.shape[1]
    x2_ = np.zeros((1, maxlen, 1)).astype('int32')
    x2_mask = np.zeros((1, maxlen, 1)).astype(theano.config.floatX)
    x2_[0,:,0] = x2
    x2_mask[0,:,0] = m2
    x2_mask = np.asarray(x2_mask, dtype=config.floatX)

    score = model.scoring_function(x1,m1,x2_,x2_mask)
    return score[0]

def evaluate(model,words,file,params):
    with open(file) as f:
        lines = f.readlines()
    result = []
    for line in lines:
        line  = line.split("\t")
        score = predictSimilarity(model, line[0], line[1], words)
        result.append([line[0], line[1], score])
    return result

def evaluate_all(model,words):
    raise NotImplementedError
    
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
    def overall_training_time(self):
        return np.sum(self.times)