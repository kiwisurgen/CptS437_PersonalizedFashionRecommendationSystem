from preprocessing.preprocess_product_data import preprocess_fashion_data
from processing.tfidf_title_similarity import tfidf_cosine_sim
from processing.onehot_generic_similarity import onehot_cosine_sim
from processing.hybrid_similarity import hybrid_similarity
from processing.price_similarity import price_similarity
from processing.rating_similarity import rating_similarity
import random

def main():
    data_path = "data/products.csv"
    df = preprocess_fashion_data(data_path)
    products = df['title'].tolist()
    categories=df['category'].tolist()
    brands=df['brand'].tolist()
    prices=df['price'].tolist()
    ratings=df['rating'].tolist()

    N = 5
    IDX = random.randint(0, len(products)-1) # use single number for testing weights

    # sim_scores = tfidf_cosine_sim(idx=IDX, n=N, products=products)
    # sim_scores = onehot_cosine_sim(idx=IDX, n=N, items=categories)
    # sim_scores = onehot_cosine_sim(idx=IDX, n=N, items=brands)
    # sim_scores = price_similarity(idx=IDX, n=N, prices=prices, alpha=3.0, normalize=True)
    # sim_scores = rating_similarity(idx=IDX, n=N, ratings=ratings, alpha=2.0, normalize=True)

    sim_scores = hybrid_similarity(
        idx=IDX, n=N, products=products,
        categories=categories, brands=brands,
        prices=prices, ratings=ratings,
        w_tfidf=0.3, w_cat=0.2, w_brand=0.3,
        w_price=0.1, w_rating=0.1
    )

    print(f"Top {N} products similar to '{products[IDX]}':")
    # print(sim_scores[0]) # debug
    for i, score in sim_scores:
        print(f" - {products[i]} (Similarity: {score:.4f})")

if __name__ == "__main__":
    main()
