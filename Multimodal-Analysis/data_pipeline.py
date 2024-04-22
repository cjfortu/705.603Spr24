import os
import pickle
from collections import deque
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import contractions
import string

from sklearn.model_selection import train_test_split


class NSTEL_Pipeline:
    """
    A class to normalize, stem, tokenize, encode, and load data for machine learning.
    """
    def __init__(self):
        self.text = None
        self.tokentext = None
        self.seq = None
        self.arrsz = 100
        with open('./models/tokshash.bin', mode='rb') as f:
            self.tokshash = pickle.load(f)
        with open('./models/dfproc.bin', mode='rb') as f:
            self.dfproc = pickle.load(f)
        
    
    def loadbatch(self):
        """
        Get the preprocessed, normalized, stemmed, tokenized, semantic encoded, and
        one hot encoded dataframe.
        """
        X = np.stack(np.array(self.dfproc['seqs']))
        Y = np.stack(np.array(self.dfproc['ratings']))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                            test_size=0.1, stratify=Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                            test_size=1/9, stratify=Y_train)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
        
        
    def preprocess(self):
        """
        Get a normalized, stemmed, tokenized form of text.
        """
        ## STRIP NUMBERS
        self.text = re.sub(r'\d', '', self.text)

        # EXPAND CONTRACTIONS
        exp_words = []    
        for word in self.text.split():
            exp_words.append(contractions.fix(word))   

        # REMOVE PUNCTUATION
        puncstr = ' '.join(exp_words)
        puncstr = puncstr.translate(str.maketrans('', '', string.punctuation))

        # TOKENIZE
        word_tokens = word_tokenize(puncstr.lower())

        # REMOVE STOPWORDS
        stop_words = set(stopwords.words('english'))
        tokentext = [w for w in word_tokens if not w.lower() in stop_words]

        # STEMMING
        stemmer = SnowballStemmer(language='english')
        self.tokentext = [stemmer.stem(token) for token in tokentext]
        
    
    def encodesingle(self):
        """
        Encode a tokenized corpus according to the existing word2vec indices.
        """
        seq = deque()
        for i in range(0, self.arrsz):
            if i < len(self.tokentext):
                try:
                # O(N) lookup for dict versus lists.  Critical for speed.
                    seq.append(self.tokshash[self.tokentext[i]])
                except:
                    seq.append(0)
            else:
                seq.append(0)

        self.seq = np.array(seq)
    
    
    def loadsingle(self, text):
        """
        Get the normalized/stemed/tokenized/encoded review.
        
        parameters:
        text (str): The text string
        
        returns:
        seq (np.array): The token indices
        """
        self.text = text
        self.preprocess()
        self.encodesingle()
        
        return self.seq