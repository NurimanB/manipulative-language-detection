# Manipulative Language Detection: Dark Patterns

Detecting manipulative language in online shopping and advertising text: first **manipulative vs. neutral** (binary), then **which technique** it uses (urgency, scarcity, social proof, emotional pressure).

Models are trained on website dark-pattern text (Mathur et al. 2019 + Yada et al. 2022) and also tested on our own set of real and synthetic advertisements to measure how well they transfer to a different domain.

## Experiments

| # | Notebook | What it does |
|---|---|---|
| 1 | [`01_baseline_tfidf_logreg.ipynb`](notebooks/01_baseline_tfidf_logreg.ipynb) | TF-IDF + Logistic Regression baseline, error analysis, group-aware CV, external checks |
| 2 | [`02_multilabel_techniques.ipynb`](notebooks/02_multilabel_techniques.ipynb) | Technique classification with One-vs-Rest Logistic Regression |
| 3 | [`03_word2vec_glove.ipynb`](notebooks/03_word2vec_glove.ipynb) | Dense representations: project-trained Word2Vec vs. pretrained GloVe |
| 4 | [`04_neural_net_backprop.ipynb`](notebooks/04_neural_net_backprop.ipynb) | One-hidden-layer neural network in NumPy with hand-derived backprop, gradient check, activation comparison, embedding fine-tuning |

All notebooks use the **same data, near-duplicate template groups, and held-out test split**, defined once in [`src/data.py`](src/data.py). Near-duplicate texts (same template up to numbers) never appear on both sides of a split.

## Results

| Model | Binary F1, held-out test | Binary F1, 5-fold group CV | Technique Macro F1, CV | Binary F1, our real ads (n=57) | Binary F1, our synthetic set (n=370) | Accuracy, hard negatives (n=90) |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.932 | **0.946** | **0.943** | 0.468 | 0.414 | **0.822** |
| Word2Vec (project) + LR | **0.940** | 0.926 | 0.880 | **0.571** | **0.498** | 0.778 |
| GloVe + LR | 0.904 | 0.896 | 0.810 | 0.415 | 0.404 | 0.800 |
| Neural net (Exp. 4) | *to be updated* | | | | | |

**Main finding so far:** TF-IDF is strongest inside the dark-patterns domain, Word2Vec transfers best to our own ads, and every model drops sharply out of domain (≈0.95 → ≈0.4–0.6 F1). The training data has no examples of fear, exaggeration, or authority, which our ads use.

## Repository layout

```
├── data/          # CSVs go here (see data/README.md)
├── notebooks/     # one notebook per experiment
├── results/       # metric tables saved by each notebook, read by later ones
└── src/
    ├── data.py        # loading, template groups, shared train/test split, save/load results
    └── embeddings.py  # tokenizer and averaged word-vector features
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the two CSVs in data/ (see data/README.md)
jupyter lab
```

Run the notebooks **in order (01 → 04)**: notebooks 3 and 4 compare against results that earlier notebooks save to `results/`. The first run of notebook 3 downloads GloVe (`glove-wiki-gigaword-100`, about 130 MB) through `gensim`.

## References
- Mathur, A. et al. (2019). *Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites.* CSCW.
- Yada, Y. et al. (2022). *Dark patterns in e-commerce: a dataset and its baseline evaluations.* IEEE BigData.
