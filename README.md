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
| 5 | [`05_language_models_rnn.ipynb`](notebooks/05_language_models_rnn.ipynb) | Language models and RNNs in NumPy: n-gram LMs (sparsity, smoothing, backoff), perplexity, text generation, RNN-LM with teacher forcing, BPTT with gradient check, RNN classifier, vanishing/exploding gradients and clipping, LM-based classification |

All notebooks use the **same data, near-duplicate template groups, and held-out test split**, defined once in [`src/data.py`](src/data.py). Near-duplicate texts (same template up to numbers) never appear on both sides of a split.

## Results

Binary classification (manipulative vs. neutral) on the shared held-out test set (470 texts). Neural models: mean over 3 seeds.

| Model | Experiment | Test F1 |
|---|---|---|
| TF-IDF + logistic regression | 1 | 0.932 |
| Word2Vec (project corpus) + logistic regression | 3 | 0.940 |
| GloVe + logistic regression | 3 | 0.904 |
| Neural network, ReLU, fine-tuned GloVe | 4 | 0.935 |
| Unigram LM classifier (= Naive Bayes) | 5 | 0.949 |
| RNN classifier, mean of hidden states | 5 | 0.947 |
| **RNN-LM classifier** (one RNN language model per class + Bayes' rule) | 5 | **0.958** |

Language-model perplexity on the test set (lower is better): best n-gram 29.2 (manipulative) / 89.8 (neutral), RNN-LM **20.3 / 67.3**.

Cross-validation and our own ads (Experiments 1–3). CV columns are from this run; the external columns are from the earlier run with `manipulative_language_dataset_300.csv`, which is not in the repo yet:

| Model | Binary F1, 5-fold group CV | Technique Macro F1, CV | Binary F1, our real ads (n=57) | Binary F1, our synthetic set (n=370) | Accuracy, hard negatives (n=90) |
|---|---|---|---|---|---|
| TF-IDF + LR | **0.946** | **0.944** | 0.468 | 0.414 | **0.822** |
| Word2Vec (project) + LR | 0.926 | 0.882 | **0.571** | **0.498** | 0.778 |
| GloVe + LR | 0.896 | 0.805 | 0.415 | 0.404 | 0.800 |

**Main findings so far**
- In-domain, treating manipulative and neutral text as two "languages" works best: class-conditional language models beat every discriminative model, and even Naive Bayes beats Experiments 1–4.
- Word order adds little for discriminative classifiers in this templated domain. A last-state RNN forgets a cue after about 10 following words (vanishing gradients), while mean pooling keeps it.
- Out of domain, every model tested so far drops sharply (≈0.95 → ≈0.4–0.6 F1). The training data has no examples of fear, exaggeration or authority, which our ads use. Experiments 4–5 still need to be evaluated on the external sets (see below).

> **Correction to Experiment 4.** In the original combined notebook, a cross-validation loop in Experiment 3 overwrote `train_idx`, so Experiment 4 was trained on a set that overlapped the test set. The test F1 of the best network was reported as 0.965; after the fix it is **0.935**. The split notebooks use the shared split from `src/data.py`.

## Repository layout

```
├── data/
│   ├── dark_patterns_ecommerce_dataset.csv   # main dataset
│   └── README.md                             # sources, licenses, columns; where the external set goes
├── notebooks/        # one notebook per experiment (01 → 05), committed with outputs
├── results/          # metric tables saved by each notebook, read by later ones
├── scripts/
│   └── build_dataset.py                      # rebuilds the main dataset from the two public sources
└── src/
    ├── data.py        # loading, template groups, shared train/test split, save/load results
    └── embeddings.py  # tokenizer and averaged word-vector features
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks **in order (01 → 05)**: later notebooks compare against results that earlier ones save to `results/`. The first run of notebook 3 downloads GloVe (`glove-wiki-gigaword-100`, about 130 MB) through `gensim`. Running all five takes roughly 20–30 minutes on a laptop CPU, most of it in notebook 5.

**External test set.** Put `manipulative_language_dataset_300.csv` in `data/` to enable the checks on our real ads and synthetic examples. Without it, those sections are skipped with a message, and everything else runs.

## References
- Mathur, A. et al. (2019). *Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites.* CSCW.
- Yada, Y. et al. (2022). *Dark patterns in e-commerce: a dataset and its baseline evaluations.* IEEE BigData.
