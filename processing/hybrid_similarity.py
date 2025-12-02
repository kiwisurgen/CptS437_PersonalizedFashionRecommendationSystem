from processing.tfidf_title_similarity import tfidf_cosine_sim
from processing.onehot_generic_similarity import onehot_cosine_sim
from processing.price_similarity import price_similarity
from processing.rating_similarity import rating_similarity

def hybrid_similarity(idx, n, products, categories, brands, prices, ratings, w_tfidf=0.2, w_cat=0.2, w_brand=0.2, w_price=0.2, w_rating=0.2):
    # similarity scores from individual methods
    tfidf_scores = tfidf_cosine_sim(idx=idx, n=None, products=products)
    cat_scores   = onehot_cosine_sim(idx=idx, n=None, items=categories)
    brand_scores = onehot_cosine_sim(idx=idx, n=None, items=brands)
    price_scores = price_similarity(idx=idx, n=None, prices=prices, alpha=3.0, normalize=True)
    rating_scores = rating_similarity(idx=idx, n=None, ratings=ratings, alpha=2.0, normalize=True)

    # create dictionaries with the scores
    tfidf_dict = dict(tfidf_scores)
    cat_dict   = dict(cat_scores)
    brand_dict = dict(brand_scores)
    price_dict = dict(price_scores)
    rating_dict = dict(rating_scores)

    hybrid_scores = []

    num_items = len(products)
    for j in range(num_items):
        if j == idx:
            continue

        score = (
            w_tfidf * tfidf_dict.get(j, 0.0) +
            w_cat   * cat_dict.get(j, 0.0) +
            w_brand * brand_dict.get(j, 0.0) +
            w_price * price_dict.get(j, 0.0) +
            w_rating * rating_dict.get(j, 0.0)
        )

        hybrid_scores.append((j, score))
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)

    if n is None:
        return hybrid_scores
    return hybrid_scores[:n]