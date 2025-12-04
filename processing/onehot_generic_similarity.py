import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def top_n_similar(idx, n=5, cosine_sim=None):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:n+1] # skip self

def top_similar(idx, cosine_sim=None):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:] # skip self

def onehot_cosine_sim(idx, n, items: list):
    onehot = pd.get_dummies(items).values # get one-hot encoded matrix
    cosine_sim = cosine_similarity(onehot, onehot) # cosine similarity between one-hot vectors
    # select top-n similar items
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if n == None:
        sim_scores = top_similar(idx, cosine_sim=cosine_sim)
    else:
        sim_scores = top_n_similar(idx=idx, n=n, cosine_sim=cosine_sim)
    return sim_scores
