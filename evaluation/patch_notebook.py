"""Patch the evaluation notebook to add robust imports and placeholders.

This script inserts a small code cell at the top of the notebook to ensure
`pandas`, `numpy`, `matplotlib.pyplot` are imported and that common variables
(`results_df`, `faiss_results_df`, `df`, `test_queries`) exist to avoid
NameError during headless execution.
"""
from pathlib import Path
import nbformat


def patch(nb_path: Path):
    nb = nbformat.read(str(nb_path), as_version=4)
    preamble = '''import pandas as pd
import numpy as np
import os
# ensure working directory
os.chdir(r"%s")

# Guard plotting imports for headless environments
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# Attempt to import evaluation helpers (may not be available in minimal env)
try:
    from evaluation.baselines import RandomRecommender, PopularityRecommender, TFIDFRecommender
    from evaluation.ann_indexing import FAISSIndex, benchmark_index, print_benchmark_results
    HELPERS_AVAILABLE = True
except Exception:
    HELPERS_AVAILABLE = False

# Placeholders to avoid NameError during automated execution
if 'results_df' not in globals():
    results_df = pd.DataFrame()
if 'faiss_results_df' not in globals():
    faiss_results_df = pd.DataFrame()
if 'df' not in globals():
    try:
        df = pd.read_csv('data/products.csv')
    except Exception:
        df = pd.DataFrame()
if 'test_queries' not in globals():
    test_queries = []
'''
    preamble_cell = nbformat.v4.new_code_cell(preamble % nb_path.parent)
    # Insert after any existing notebook title cell (put at index 1 if first cell is markdown title)
    insert_at = 0
    if nb.cells and nb.cells[0].cell_type == 'markdown':
        insert_at = 1
    nb.cells.insert(insert_at, preamble_cell)
    nbformat.write(nb, str(nb_path))


if __name__ == '__main__':
    nb_path = Path(__file__).resolve().parent.parent / 'evaluation_benchmark.ipynb'
    print(f"Patching notebook: {nb_path}")
    patch(nb_path)
    print("Patched.")
