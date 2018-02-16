import lasagne
from yaml import load, YAMLError
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def str2learner(v):
    if v is None:
        return lasagne.updates.adagrad
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('@params: A type that was supposed to be a learner is not.')

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('@params: A type that was supposed to be boolean is not boolean.')


class params(object):
    
    def __init__(self):
        self.LW = 1000
        self.eta = 0.01
        self.hyper_k1 = 10000000
        self.hyper_k2 = 10000000
        self.margin=1
        self.siamese = True
        self.NEG_SAMPLE_NUM = 7
        self.epochs = 15

    def __str__(self):
        return '\n'.join(['{} : {}'.format(k, v) for k, v in self.__dict__.items()])

    def load_from_yaml(self,d='main/params.yaml'):
        with open(d, 'r') as stream:
            try:
                _params = load(stream, Loader=Loader)
                self.wordfile  = _params['wordfile']
                self.LW        = _params['LW']
                self.eta       = _params['eta']
                self.hyper_k1  = _params['hyper_k1']
                self.hyper_k2  = _params['hyper_k2']
                self.train     = _params['train']
                self.epochs    = _params['epochs']
                self.learner   = str2learner(_params['learner'])
                self.batchsize = _params['batchsize']
                self.clip      = _params['clip']
                self.save      = str2bool(_params['save'])
                self.outfile   = _params['outfile']
                self.evaluate  = str2bool(_params['evaluate'])
                self.terms_of_ontology_1 = _params['terms_of_ontology_1']
                self.terms_of_ontology_2 = _params['terms_of_ontology_2']
                self.ground_truth_alignments = _params['ground_truth_alignments']
            except YAMLError as exc:
                print(exc)