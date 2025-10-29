import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TESTING (REMOVE LATER)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.preprocess_product_data import preprocess_fashion_data
# TESTING (REMOVE LATER)

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

# TESTING (REMOVE LATER)
def main():
    data_path = "data/products.csv"
    df = preprocess_fashion_data(data_path)
    products = df['title'].tolist()
    N = 5
    IDX = 23
    sim_scores = tfidf_cosine_sim(idx=IDX, n=N, products=products)
    print(f"Top {N} products similar to '{products[IDX]}':")
    for i, score in sim_scores:
        print(f" - {products[i]} (Similarity: {score:.4f})")

if __name__ == "__main__":
    main()
# TESTING (REMOVE LATER)
