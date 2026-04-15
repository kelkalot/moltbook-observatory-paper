# The Moltbook Observatory Archive: Companion Code

[![Paper](https://img.shields.io/badge/Paper-Nature%20Scientific%20Data-blue)](ARXIV link here)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the analysis code and figure generation scripts for the paper:

> **The Moltbook Observatory Archive: an incremental dataset of agent-only social network activity**
>
> Sushant Gautam, Klas Petterson, Michael A. Riegler
>
>
> LINK: 

## Overview

[Moltbook](https://moltbook.com) is a social media platform where posts and comments are authored exclusively by autonomous AI agents. The **Moltbook Observatory Archive** is a continuously growing dataset that passively records agent profiles, posts, comments, community metadata, and platform-level metrics by polling the Moltbook API.

This companion repository provides:

- **`moltbook/`** — Analysis toolkit (risk detection, sentiment, agent scoring, network analysis, near-duplicate detection)
- **`run_analysis.py`** — End-to-end analysis pipeline producing HTML reports with embedded visualizations
- **`generate_paper_figures.py`** — Script to reproduce all publication figures

## Dataset

The dataset is hosted on the Hugging Face Hub:

```
https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
```

The snapshot described in the paper (`2026-04-15`) contains:

| Table | Rows |
|-------|------|
| agents | 175,886 |
| posts | 2,615,098 |
| comments | 1,213,007 |
| submolts | 6,730 |
| snapshots | ~1,800 |
| word_frequency | ~200,000 |

## Getting Started

### Requirements

- Python ≥ 3.9

### Installation

```bash
git clone https://github.com/kelkalot/moltbook-observatory-paper.git
cd moltbook-observatory-paper
pip install -r requirements.txt
```

### Reproduce the Analysis

Run the full analysis pipeline (approximately 90 minutes on a modern laptop):

```bash
python run_analysis.py
```

This produces:
- A timestamped JSON report in `reports/`
- An HTML report with embedded charts
- A comparison report if previous reports exist

### Reproduce the Figures

Generate all six publication figures:

```bash
python generate_paper_figures.py
```

Figures are saved as PDF files in `paper_figures/`.

### Load the Dataset

```python
from datasets import load_dataset

# Load posts
posts = load_dataset("SimulaMet/moltbook-observatory-archive",
                     "posts", split="archive")

# Load comments
comments = load_dataset("SimulaMet/moltbook-observatory-archive",
                        "comments", split="archive")
```

Or use the built-in loader:

```python
from moltbook.data import load_from_huggingface

posts_df, comments_df = load_from_huggingface()
```

## Analysis Toolkit Modules

| Module | Description |
|--------|-------------|
| `moltbook/data.py` | Data loading from HuggingFace Hub |
| `moltbook/risk.py` | Safety indicator detection (prompt injection, crypto, spam, manipulation) |
| `moltbook/sentiment.py` | Composite sentiment analysis (VADER + TextBlob) |
| `moltbook/agent_score.py` | Per-agent composite risk scoring (8-component weighted sum) |
| `moltbook/network.py` | Reply-graph construction and community detection |
| `moltbook/similarity.py` | Near-duplicate detection via MinHash/LSH |
| `moltbook/temporal.py` | Temporal pattern analysis (daily, hourly distributions) |
| `moltbook/engagement.py` | Engagement quality metrics (organic, bot, self-interaction) |
| `moltbook/clustering.py` | Content clustering via TF-IDF and K-Means |
| `moltbook/visualizations.py` | Chart generation for reports |
| `moltbook/report.py` | HTML report generation |

## Related Resources

- **Moltbook Observatory** (collector & dashboard): [github.com/kelkalot/moltbook-observatory](https://github.com/kelkalot/moltbook-observatory)
- **Dataset on HuggingFace**: [SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)
- **Risk Assessment Report**: [Zenodo DOI: 10.5281/zenodo.18444900](https://doi.org/10.5281/zenodo.18444900)

## Citation

If you use this code or the dataset in your research, please cite:

```bibtex
@article{gautam2026moltbook,
  title     = {The Moltbook Observatory Archive: an incremental dataset of
               agent-only social network activity},
  author    = {Gautam, Sushant and Petterson, Klas and Riegler, Michael A.},
  journal   = {arxiv},
  year      = {2026},
  doi       = {PLACEHOLDER_DOI},
  url       = {PLACEHOLDER_PAPER_URL}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
