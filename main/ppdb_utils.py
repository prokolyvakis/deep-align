from utils import *
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances


def getTargets(batch_len, neg_num=3):
  # The targets are always the same. We always want the positive elements of
  # the softmax (so index 0 (and possibly 1)) to be maximum (so 1 (or .5 if we
  # have two)).
    targets = np.zeros((batch_len, neg_num), dtype=np.float32)
    targets[:,[0,0]] = 1.0

    return targets

def createAnts(syns, ants, neg_num, start=0):
    S = []
    A = []
    for syn in syns:
        S.append(syn.representation)
    for ant in ants:
        A.append(ant.representation)
    dist = cosine_distances( np.array(S), np.array(A) )
    
    sort_dist = np.argsort( dist )
    
    res= {}
    
    for idx, syn in enumerate(syns):
        tmp = []
        for i in range(start,start+neg_num):
            tmp.append(ants[sort_dist[idx,i]].embeddings)
        res[syn.phrase] = tmp    
    return res

def getBatchPairs(model, batch, params, ants):
    
    g = []
    neg_num = params.NEG_SAMPLE_NUM
    batchsize = params.batchsize

    negs=[[] for x in range(batchsize)]; lengths = []

    for idx, e in enumerate(batch):
        g.append(e[0].embeddings)
        tag = ants[e[0].phrase]

        negs[idx].append(e[1].embeddings)
        negs[idx] += tag
        lengths.append(list(map(len, negs[idx])))

    maxlen = np.max(lengths)
    gx, gmask = prepare_data(g, max_len=maxlen)
    maxlen = max(gmask.shape[1], maxlen)
   
    x = np.zeros((batchsize, maxlen, neg_num+1)).astype('int32')
    x_mask = np.zeros((batchsize, maxlen, neg_num+1)).astype(theano.config.floatX)
    for batch_idx, neg in enumerate(negs):
        for idx, s in enumerate(neg):
            x[batch_idx, 0:lengths[batch_idx][idx], idx] = s
            x_mask[batch_idx, 0:lengths[batch_idx][idx], idx] = 1.
    x_mask = np.asarray(x_mask, dtype=config.floatX)
    targets = np.asarray(getTargets(len(batch), neg_num+1), dtype=config.floatX)

    return (gx, gmask, x, x_mask, targets)





def train(model, data, words, params, synonyms=None, antonyms=None, start=0):
    start_time = time()

    ants = createAnts(synonyms, antonyms, params.NEG_SAMPLE_NUM, start=start)


    counter = 0
    try:
        for eidx in xrange(params.epochs):

            kf = list(get_minibatches_idx(len(data), params.batchsize, shuffle=True))
            uidx = 0
            for _, train_index in kf:

                uidx += 1

                batch = [data[t] for t in train_index]
                for i in batch:
                    i[0].populate_embeddings(words)
                    i[1].populate_embeddings(words)
                (g1x, g1mask, train_data, train_mask, targets) = getBatchPairs(model, batch, params, ants)

                cost = model.train_function(g1x, g1mask, train_data, train_mask, targets)

                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected')

                if (checkIfQuarter(uidx, len(kf))):
                    if (params.save):
                        counter += 1
                        saveParams(model, params.outfile + str(counter) + '.pickle')
                    if (params.evaluate):
                        evaluate_all(model, words)
                        sys.stdout.flush()

                #undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    i[0].unpopulate_embeddings()
                    i[1].unpopulate_embeddings()

            if (params.save):
                counter += 1
                saveParams(model, params.outfile + str(counter) + '.pickle')

            if (params.evaluate):
                evaluate_all(model, words)

            print('Epoch ', (eidx + 1), 'Cost ', cost)

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time()
    print("total time:", (end_time - start_time))