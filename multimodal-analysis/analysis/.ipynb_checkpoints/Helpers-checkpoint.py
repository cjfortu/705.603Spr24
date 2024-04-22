from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import contractions
import string
import re
from collections import deque


def preprocess(text, rating):
    """
    Normalize a text string.
    
    used by:
    get_proccorpus()
    
    parameters:
    text (str): The text string
    
    returns:
    tokentext (list of str): The processed text as separate tokens in their own string
    proctext (list of str): The processed text as joined tokens in a single string
    rating (float): The associated star rating
    """
    ## STRIP NUMBERS
    text = re.sub(r'\d', '', text)
    
    # EXPAND CONTRACTIONS
    exp_words = []    
    for word in text.split():
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
    tokentext = [stemmer.stem(token) for token in tokentext]
    
    proctext = ' '.join(tokentext)
    
    return (tokentext, proctext, rating)


def tokentext2seqs(tokentext, tokshash, arrsz):
    """
    Convert tokens to their indices in the word2vec model.
    
    parameters:
    ttxt (list of str): The tokens
    tokshash (dict of int): The word2vec indices of each token
    inputsz (int): The size limit of the input vectors
    
    returns:
    seq (numpy array): The indices of the tokens
    """
    seq = deque()
    for i in range(0, arrsz):
        if i < len(tokentext):
            # O(N) lookup for dict versus lists.  Critical for speed.
            seq.append(tokshash[tokentext[i]])
        else:
            seq.append(0)
    
    seq = np.array(seq)
    
    return seq