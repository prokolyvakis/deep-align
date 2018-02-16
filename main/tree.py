def lookupIDX(words,w):
        w = w.lower()
        if w in words:
            return words[w]
        else:
            return words['something']

def addOOVwords(examples, words, We, mean=0, sigma=0.01):
    import numpy as np

    dim = We[0].shape[0]
    next_idx = len(words)
    We_OOV = []


    for example in examples:
        for ex in [example[0],example[1]]:
            phrase = ex.phrase.lower()
            listOfWords = phrase.split()
            for word in listOfWords:
                if word not in words:
                    words[word] = next_idx
                    We_OOV.append(np.random.normal(mean, sigma, dim))
                    next_idx+=1
    We_OOV = np.array(We_OOV)
    We = np.concatenate((We, We_OOV), axis=0)

    return We

class tree(object):

    def __init__(self, phrase, words):
        self.phrase = phrase
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            self.embeddings.append(lookupIDX(words,i))

    def unpopulate_embeddings(self):
        self.embeddings = []