from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import utils
from evaluate import getSeq, predictSimilarity
from match import Matcher
import re

def preferances(matrix):
    res = {}
    for x, l in enumerate(matrix):
        tmp = [i[0] for i in sorted(enumerate(l), key=lambda x:x[1])]
        res[x] = tmp
    return res

def ontology_alignment(model, ontoTerms_a, ontoTerms_b, words, ceil = 0.5):

    with open(ontoTerms_a) as f:
        ontoText_a = f.readlines()
    with open(ontoTerms_b) as f:
        ontoText_b = f.readlines()
    # Remove whitespace characters like `\n` at the end of each line.
    ontoText_a = [x.strip() for x in ontoText_a] 
    ontoText_b = [x.strip() for x in ontoText_b]

    whole = []
    for text_a in ontoText_a:
        for text_b in ontoText_b:
            txt_a = re.sub(' +',' ',text_a)
            txt_b = re.sub(' +',' ',text_b)
            if txt_a == txt_b:
                whole.append([text_a, text_b, 0.0])
                try:
                    ontoText_a.remove(text_a)
                except ValueError:
                    print(text_a)
                try:
                    ontoText_b.remove(text_b)
                except ValueError:
                    pass
                    #print(text_b)
    # Transform to Word & Mask vectors to apply "feedforward_function"
    ontoData_a, ontoData_b = [], []
    for sentence in ontoText_a:
        ontoData_a.append(getSeq(sentence, words))
    for sentence in ontoText_b:
        ontoData_b.append(getSeq(sentence, words))
    x1,m1 = utils.prepare_data(ontoData_a)
    x2,m2 = utils.prepare_data(ontoData_b)
    OntoEmbg_a = model.feedforward_function(x1,m1)
    OntoEmbg_b = model.feedforward_function(x2,m2)
    # Compute the Cosine Distances:
    dist = cosine_distances(OntoEmbg_a,OntoEmbg_b)
    disT = np.transpose(dist)

    
    males    = preferances(dist)
    females  = preferances(disT)
    match = Matcher(males, females)
    marriage = match()

    for key, value in marriage.items():
        man         = ontoText_a[value]
        woman       = ontoText_b[key]
        value       = dist[value][key]
        if value < ceil:
            whole.append([man, woman, value])
    return whole


def alignment_evaluation(model, words, alignments, ground_truth, choice='1-1'):
    WARNING = ' \x1b[31m'
    ENDC =  '\x1b[0m'
    
    numOfAllignments1vs1 = len(cluster_m2n_maps(ground_truth))

    with open(ground_truth) as f:
        data = f.readlines()
    lines = [x.strip() for x in data]

    truth = set()
    for line in lines:
        line  = line.split("\t")
        truth.add((line[0],line[1]))

    found = set()
    cnt = 0; noise=0;
    corrects = []; wrongs = []; whole = []
    for man, woman, value in alignments:
        msg =  '(%s, %s) -> %f' % (man, woman, value)
        if (man, woman) in truth: 
            corrects.append(msg)
            cnt+=1
            found.add((man, woman))
        elif (woman, man) in truth:
            corrects.append(msg)
            cnt+=1
            found.add((woman, man))
        else:
            msg = WARNING + msg + ENDC
            wrongs.append(msg)
            noise+=1; 
        whole.append(msg)

    not_found = list(set(truth) - set(found))
    not_found_ = []
    for sentence_1, sentence_2 in not_found:
        cost = 1 - predictSimilarity(model, sentence_1, sentence_2, words)
        msg =  '(%s, %s) -> %f' % (sentence_1, sentence_2, cost)
        not_found_.append(msg)

    cnt   = len(corrects)
    noise = len(wrongs)

    numOfAllignments = len(lines)
    precision = (1.0*cnt)/(cnt+noise)
    if choice == '1-1':
        recall    = (1.0*cnt)/numOfAllignments1vs1
    elif choice == 'm-n':
        recall    = (1.0*cnt)/numOfAllignments
    else:
        raise ValueError('alignment_evaluation: The choice variable had improper value: %s.\n' % (choice))
    print('The precision is: %f' % (precision))
    print('The recall    is: %f' % (recall))
    print('The F1-score  is  %f' % ((2.0*precision*recall)/(precision+recall)))
    print('-'*117)
    print('The number of Alignments is: %d' % (numOfAllignments1vs1 if choice == '1-1' else numOfAllignments))
    print('The number of found Alignments is: %d' % (cnt))
    print('The noise is: %d' % (noise))
    return (whole, not_found_, (numOfAllignments1vs1 if choice == '1-1' else numOfAllignments), cnt, noise)
    
def idxInMapsCluster(origin, target, cluster):
    query = (origin, target)
    for idx, setOfTuples in enumerate(cluster):
        if query in setOfTuples:
            return idx
    return -1

def readAlignmentsFromText(ground_truth):
    with open(ground_truth) as f:
        data = f.readlines()
    lines = [x.strip() for x in data]
    truth = []
    for line in lines:
        tmp  = line.split("\t")
        truth.append([tmp[0],tmp[1]])
    return truth

def cluster_m2n_maps(ground_truth_dir):
    truth = readAlignmentsFromText(ground_truth_dir)

    truth_clustered = []
    for origin, target in truth:
        flag=False
        for cluster in truth_clustered:
            for x, y in cluster:
                if origin == x or target == y:
                    cluster.append((origin, target))
                    flag = True
                    break
        if not flag:
            truth_clustered.append([(origin,target)]) 
            
    truth_clustered_new = []       
    for cluster in truth_clustered:
        tmp = set(cluster)
        truth_clustered_new.append(tmp)
    return truth_clustered_new