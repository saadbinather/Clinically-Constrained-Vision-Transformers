# Modular chest X-ray experiments

This project reproduces the notebook's sampling, preprocessing, architectures, hyperparameters, and seed-42 80/20 splits. It supports Apple Silicon MPS automatically and falls back to CPU.

Install with `python3 -m pip install -r requirements.txt`, place the three notebook datasets under `nih_data/`, `chexpert_data/`, and `rsna_data/`, then run:

```bash
python3 train.py --data-root . --out runs
```

Run one configuration with `--only nih_densenet121`. Each run saves `config.json`, split manifests, `training_log.json`, and `best.pth`; `runs/results.csv` is the consolidated report.

Protocol warning: the notebook uses the same 20% holdout for checkpoint selection and final reporting (NIH calls it `nih_test_loader`, CheXpert calls it `chex_test_loader`). This implementation preserves that behavior and labels the result accordingly; it is not an uncontaminated test evaluation. The notebook's original NIH `sample(n=10000)` is overwritten later by the full pneumonia/3x-negative construction, so this implementation preserves the effective final dataset.
