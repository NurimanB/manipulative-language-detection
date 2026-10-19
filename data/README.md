# Data

| File | Role | In the repo? |
|---|---|---|
| `dark_patterns_ecommerce_dataset.csv` | Main dataset (training + internal evaluation), 2,349 texts | No (`data/*.csv` is git-ignored). **Rebuild it** with the command below before running the notebooks |
| `manipulative_language_dataset_300.csv` | External test set only: our real ads (57) and synthetic examples (370, incl. 90 hard negatives) | **Add it here.** Without it, the notebooks skip the external checks |

## Rebuilding the main dataset

`build_dataset.py` (in the repository root) downloads the two public sources and writes the CSV to the current folder:

```bash
cd data
python ../build_dataset.py
```

It combines:
- **Mathur et al. (2019), "Dark Patterns at Scale"**: 1,818 dark-pattern texts from shopping websites with a Pattern Type (GPL-3.0, https://github.com/aruneshmathur/dark-patterns)
- **Yada et al. (2022), "Dark patterns in e-commerce: a dataset and its baseline evaluations"**: the deduplicated Mathur texts plus the same number of non-dark-pattern texts from the same websites (Apache-2.0, https://github.com/yamanalab/ec-darkpattern)

Because the CSV contains the Mathur texts, the GPL-3.0 terms apply when it is redistributed.

## Columns
- Main: `id`, `text`, `manipulative` (0/1), `urgency`, `scarcity`, `social_proof`, `emotional_pressure`, `pattern_category`, `pattern_type`, `page_id`, `source`
- External: `id`, `text`, `manipulative`, the same four technique columns, `fear`, `exaggeration`, `authority` (not used: the main dataset has no such labels), `confidence`, `source_type` (`real` / `synthetic`), `source` (e.g. `synthetic_hard_negative`)
