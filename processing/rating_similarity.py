import numpy as np

def rating_similarity(idx, n, ratings, alpha=2.0, normalize=True):
    ratings_arr = np.array(ratings, dtype=float)

    # normalize ratings to range 0 - 1
    if normalize:
        r_min = ratings_arr.min()
        r_max = ratings_arr.max()
        if r_max > r_min:
            norm_ratings = (ratings_arr - r_min) / (r_max - r_min)
        else:
            norm_ratings = np.zeros_like(ratings_arr) # all ratings identical error case
    else:
        norm_ratings = ratings_arr

    # distance from the target rating
    target_rating = norm_ratings[idx]
    diffs = np.abs(norm_ratings - target_rating)

    # use linear decay
    # sims = 1 - np.clip(diffs / diffs.max(), 0, 1)
    # or convert distance to similarity using exponential decay
    sims = np.exp(-alpha * diffs)

    sim_scores = list(enumerate(sims))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:] # skip self

    if n is None:
        return sim_scores
    
    return sim_scores[:n]
