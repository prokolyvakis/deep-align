from collections import defaultdict

class Matcher:

    def __init__(self, men, women):
        '''
        Constructs a Matcher instance.
        Takes a dict of men's spousal preferences, `men`,
        and a dict of women's spousal preferences, `women`.
        '''
        self.M = men
        self.W = women
        self.M_size = len(men)
        self.W_size = len(women)
        self.wives = {}
        self.pairs = []

        # we index spousal preferences at initialization 
        # to avoid expensive lookups when matching
        self.mrank = defaultdict(dict)  # `mrank[m][w]` is m's ranking of w
        self.wrank = defaultdict(dict)  # `wrank[w][m]` is w's ranking of m

        for m, prefs in men.items():
            for i, w in enumerate(prefs):
                self.mrank[m][w] = i

        for w, prefs in women.items():
            for i, m in enumerate(prefs):
                self.wrank[w][m] = i


    def __call__(self):
        return self.match()

    def prefers(self, w, m, h):
        '''Test whether w prefers m over h.'''
        return self.wrank[w][m] < self.wrank[w][h]

    def after(self, m, w):
        '''Return the woman favored by m after w.'''
        i = self.mrank[m][w] + 1    # index of woman following w in list of prefs
        if i < self.W_size:
            return self.M[m][i]
        return None                 # No wives left for man "m"
        

    def match(self):
        '''
        Try to match all men with their next preferred spouse.
        
        '''
        men = list(self.M.keys())         # get the complete list of men
        
        # Map each man to their first preference
        next_ = dict((m, rank[0]) for m, rank in self.M.items()) 
        
        wives = {}                  # mapping from women to current spouse
        
        while men:
            m, men = men[0], men[1:]
            w = next_[m]                        # next woman for m to propose to
            # Check if there are still possible wives, control needed for unequal sets!
            if w:                               
                next_[m] = self.after(m, w)     # woman after w in m's list of prefs
                if w in wives:
                    h = wives[w]                # current husband
                    if self.prefers(w, m, h):
                        men.append(h)           # husband becomes available again
                        wives[w] = m            # w becomes wife of m
                    else:
                        men.append(m)           # m remains unmarried
                else:
                    wives[w] = m                # w becomes wife of m
                    
        self.pairs = [(h, w) for w, h in wives.items()]
        self.wives = wives
        return wives
        #return self.match(men, next_, wives)