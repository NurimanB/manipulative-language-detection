# Data

The notebooks expect two CSV files in this folder.

| File | Role | Source |
|---|---|---|
| `dark_patterns_ecommerce_dataset.csv` | Main dataset (training + internal evaluation) | Built by `build_dataset.py` from Mathur et al. (2019) and Yada et al. (2022) |
| `manipulative_language_dataset_300.csv` | External test set only (real ads + synthetic examples incl. hard negatives) | Our original project data |

## Licenses
- **Mathur et al. (2019), "Dark Patterns at Scale"**: GPL-3.0
- **Yada et al. (2022), "Dark patterns in e-commerce: a dataset and its baseline evaluations"**: Apache-2.0

If you redistribute the combined CSV, the GPL-3.0 terms of the Mathur data apply.

## Expected columns
- Main: `text`, `manipulative` (0/1), `source`, `pattern_type`, `urgency`, `scarcity`, `social_proof`, `emotional_pressure`
- External: `id`, `text`, `manipulative`, `source`, `source_type` (`real` / `synthetic`), and the same four technique columns
