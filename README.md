# Yeast-KO-Fitness

Code for **Held-Out-Gene Yeast KO Fitness Prediction with Genetic Interaction and Molecular Features**.

We predict the condition-specific fitness profile of an unseen *S. cerevisiae* gene knockout across 7,682 chemogenomic conditions, using Gene Ontology annotations, ESM2 protein embeddings, and genetic interaction (GI) profiles as gene features, combined through a FiLM-conditioned residual scoring head.

## Code

- `src/models.py`: MLP gene encoder, dot-product and FiLM scoring heads, masked MSE and per-gene Pearson loss.
- `src/train.py`: trains one model on a gene-wise split (3,591/448/448) and evaluates it on the held-out test genes.

## Data

`train.py` reads the following files from `--data`, with rows aligned to the same gene order:

| File | Key | Shape |
|---|---|---|
| `target.npz` | `Y` | genes × conditions fitness matrix, `NaN` = not measured |
| `feat_go.npz` | `X` | genes × GO terms, multi-hot |
| `feat_esm.npz` | `X` | genes × 1280, ESM2-650M mean embedding |
| `feat_gi.npz` | `X` | genes × (GI scores + observation mask) |

## Usage

Requires Python 3, PyTorch, NumPy and SciPy.

```bash
cd src
python train.py --data ../data --feats go gi --head film --lam 0.2 --seed 42
python train.py --data ../data --feats go esm gi --head film --lam 0.2 --seed 42
python train.py --data ../data --feats gi --head dot --lam 0 --seed 42
```

Each run writes `metrics.json` (per-gene Pearson, overall Pearson, Spearman, RMSE on the test genes) and `pred_test.npy` to `--out`.

## Status

🚧 This repository is under construction and will be updated.
