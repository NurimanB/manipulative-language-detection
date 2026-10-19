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
| 6 | [`06_attention_transformers.ipynb`](notebooks/06_attention_transformers.ipynb) | Attention and Transformers in NumPy: attention pooling for the RNN, multi-head scaled dot-product self-attention, position representations, masking, residual connections and layer normalization, Transformer encoder classifier with ablations, Transformer decoder LM, gradient checks, attention weights, long-distance and word-order tests, quadratic cost |

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
| RNN-LM classifier (one RNN language model per class + Bayes' rule) | 5 | 0.958 |
| RNN classifier, attention pooling | 6 | 0.944 |
| Transformer encoder (1 block, 4 heads, sinusoidal positions, mean pooling) | 6 | 0.953 |
| **Transformer-LM classifier** (one Transformer decoder LM per class + Bayes' rule) | 6 | **0.960** |

Language-model perplexity on the test set (lower is better), manipulative / neutral texts: best n-gram 29.2 / 89.8, RNN-LM 20.3 / 67.3, Transformer LM 20.6 / 64.9.

Long-distance test (Experiment 6): manipulative test texts that are still detected after 40 neutral words are appended: RNN with last state 19%, with mean pooling 45%, **with attention pooling 74%**.

Cross-validation (Experiments 1–3):

| Model | Binary F1, 5-fold group CV | Technique Macro F1, CV |
|---|---|---|
| TF-IDF + LR | **0.946** | **0.944** |
| Word2Vec (project) + LR | 0.926 | 0.882 |
| GloVe + LR | 0.896 | 0.805 |

Our own data (`manipulative_language_dataset_300.csv`, not in the repo), as collected in the comparison table of notebook 06. Models trained on the dark-patterns data only; neural models: 3 seeds combined.

| Model | Experiment | Binary F1, our real ads (n=57) | Binary F1, our synthetic set (n=370) | Accuracy, hard negatives (n=90) |
|---|---|---|---|---|
| TF-IDF + LR | 1 | 0.468 | 0.414 | 0.822 |
| Word2Vec (project) + LR | 3 | 0.571 | **0.498** | 0.778 |
| GloVe + LR | 3 | 0.415 | 0.404 | 0.800 |
| Neural network, ReLU, fine-tuned GloVe | 4 | 0.521 | 0.449 | 0.715 |
| Unigram LM classifier (= Naive Bayes) | 5 | 0.473 | 0.375 | 0.867 |
| Bigram LM classifier | 5 | **0.621** | 0.411 | 0.778 |
| RNN-LM classifier | 5 | 0.510 | 0.412 | **0.889** |
| RNN classifier, last state / mean of states | 5 | 0.566 / 0.511 | 0.447 / 0.402 | 0.789 / 0.833 |
| RNN classifier, attention pooling | 6 | 0.520 | 0.413 | 0.844 |
| Transformer encoder | 6 | 0.480 | 0.244 | 0.833 |
| Transformer-LM classifier | 6 | 0.536 | 0.408 | 0.800 |

**Main findings so far**
- In-domain, treating manipulative and neutral text as two "languages" works best: class-conditional language models (RNN or Transformer decoder) give the highest scores, and even Naive Bayes beats Experiments 1–4. The in-domain task is close to saturated: the best models differ by less than one standard deviation over seeds.
- Word order adds little in this templated domain. A Transformer without position vectors is exactly order-blind and still scores as well as with positions.
- Attention fixes the long-distance problem of RNNs: a last-state RNN forgets a cue after about 10 following words (vanishing gradients), mean pooling dilutes it in long texts, while attention pooling keeps focusing on it.
- Out of domain, every model drops sharply (≈0.95 → 0.42–0.62 F1 on our real ads), including the Transformers of Experiment 6. The best transfer comes from simple models (bigram LM 0.62, Word2Vec + LR 0.57), and the Transformer encoder transfers worst to our synthetic ad copy (0.24). The training data has no examples of fear, exaggeration or authority, which our ads use: the gap is in *which* cues occur, and no architecture closes it.

> **Correction to Experiment 4.** In the original combined notebook, a cross-validation loop in Experiment 3 overwrote `train_idx`, so Experiment 4 was trained on a set that overlapped the test set. The test F1 of the best network was reported as 0.965; after the fix it is **0.935**. The split notebooks use the shared split from `src/data.py`.

## Repository layout

```
├── data/
│   ├── dark_patterns_ecommerce_dataset.csv   # main dataset (git-ignored; built by build_dataset.py)
│   └── README.md                             # sources, licenses, columns; where the external set goes
├── notebooks/        # one notebook per experiment (01 → 06), committed with outputs
├── results/          # metric tables saved by each notebook, read by later ones
├── build_dataset.py  # rebuilds the main dataset from the two public sources
└── src/
    ├── data.py        # loading, template groups, shared train/test split, save/load results
    └── embeddings.py  # tokenizer and averaged word-vector features
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd data && python ../build_dataset.py)   # downloads the two public sources, writes the main CSV
jupyter lab
```

Run the notebooks **in order (01 → 06)**: later notebooks compare against results that earlier ones save to `results/`. The first run of notebook 3 downloads GloVe (`glove-wiki-gigaword-100`, about 130 MB) through `gensim`. Running all six took about 50 minutes in our last run (4-core CPU, `OMP_NUM_THREADS=1`), almost all of it in notebooks 5 and 6 (about 20 minutes each).

**External test set.** Put `manipulative_language_dataset_300.csv` in `data/` to enable the checks on our real ads and synthetic examples. Without it, those sections are skipped with a message, and everything else runs.

## References
- Mathur, A. et al. (2019). *Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites.* CSCW.
- Yada, Y. et al. (2022). *Dark patterns in e-commerce: a dataset and its baseline evaluations.* IEEE BigData.
