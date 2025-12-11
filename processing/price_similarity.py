import numpy as np

def price_similarity(idx, n, prices, alpha=3.0, normalize=True):
    prices_arr = np.array(prices, dtype=float)

    # normalize prices to range 0 - 1
    if normalize:
        p_min = prices_arr.min()
        p_max = prices_arr.max()
        if p_max > p_min:
            norm_prices = (prices_arr - p_min) / (p_max - p_min)
        else:
            norm_prices = np.zeros_like(prices_arr) # all prices identical error case
    else:
        norm_prices = prices_arr

    # distance from the target price
    target_price = norm_prices[idx]
    diffs = np.abs(norm_prices - target_price)

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
