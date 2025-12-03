"""Run the evaluation notebook and export a markdown report.

Usage:
    python evaluation/generate_report.py --notebook ../evaluation_benchmark.ipynb --output ../Documentation/EVALUATION.md

This script executes the notebook in-place and converts it to Markdown. It is safe to re-run; heavy computations in the notebook
will use existing embeddings and FAISS index when available.
"""
import argparse
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import MarkdownExporter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_notebook(notebook_path: Path, timeout: int = 600):
    logger.info(f"Loading notebook {notebook_path}")
    nb = nbformat.read(str(notebook_path), as_version=4)
    # Prepend a small preamble cell to ensure common imports and correct cwd
    preamble_code = (
        "import sys, os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "# Ensure notebook runs with repo root as working directory\n"
        f"os.chdir(r'{str(notebook_path.parent)}')\n"
    )
    preamble_cell = nbformat.v4.new_code_cell(preamble_code)
    nb.cells.insert(0, preamble_cell)

    # Remove any notebook cells that try to generate/write the final report directly
    # so we can safely execute the analysis cells and then export the full notebook.
    filtered_cells = []
    for cell in nb.cells:
        src = cell.get('source', '')
        if isinstance(src, str) and ('generate_markdown_report(' in src or 'Documentation/EVALUATION.md' in src or 'report_md =' in src):
            # skip this cell
            continue
        filtered_cells.append(cell)
    nb.cells = filtered_cells

    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3', allow_errors=True)
    logger.info("Executing notebook (this may take a while)...")
    ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
    # Save executed notebook in-place
    nbformat.write(nb, str(notebook_path))
    return nb


def notebook_to_markdown(nb, output_path: Path):
    exporter = MarkdownExporter()
    logger.info(f"Converting notebook to Markdown: {output_path}")
    (body, resources) = exporter.from_notebook_node(nb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding='utf-8')
    logger.info(f"Wrote report to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', default='evaluation_benchmark.ipynb', help='Notebook to execute (relative to repo root)')
    parser.add_argument('--output', default='Documentation/EVALUATION.md', help='Markdown output path')
    parser.add_argument('--timeout', type=int, default=600, help='Execution timeout in seconds')
    args = parser.parse_args()

    nb_path = Path(args.notebook).resolve()
    out_path = Path(args.output).resolve()

    if not nb_path.exists():
        logger.error(f"Notebook not found: {nb_path}")
        raise SystemExit(2)

    nb = run_notebook(nb_path, timeout=args.timeout)
    notebook_to_markdown(nb, out_path)


if __name__ == '__main__':
    main()
