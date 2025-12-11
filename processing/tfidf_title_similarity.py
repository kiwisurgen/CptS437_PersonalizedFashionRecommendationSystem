import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def top_n_similar(idx, n=5, cosine_sim=None):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:n+1] # skip self

def top_similar(idx, cosine_sim=None):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:] # skip self

def tfidf_cosine_sim(idx, n, products:list):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(products)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    if n == None:
        sim_scores = top_similar(idx, cosine_sim=cosine_sim)
    else:
        sim_scores = top_n_similar(idx=idx, n=n, cosine_sim=cosine_sim)
    return sim_scores
