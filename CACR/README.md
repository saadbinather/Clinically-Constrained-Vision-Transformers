# Clinical Attention Consistency Regularization

This folder contains my medical-imaging research experiments on cross-domain chest X-ray diagnosis.

## Main idea

The project uses a Vision Transformer and RSNA pneumonia bounding-box annotations to encourage the model to focus on clinically relevant regions. The model is evaluated across NIH ChestX-ray14 and CheXpert.

## Files

- `experiments.ipynb` — dataset preparation, model training, attention analysis, ablations, and evaluation.
- `src/` — source-code space for cleaned research code.
- `results/` — results and experiment outputs.

## Recorded results

- NIH subset: 5,412 images.
- CheXpert subset: 9,675 frontal images.
- RSNA subset: 13,012 unique patients with pneumonia bounding-box annotations.
- A recorded NIH-to-CheXpert baseline evaluation reports AUROC **0.5426**.
- The paper reports improvement from **0.5369 to 0.6283 AUROC at λ = 0.5**. This is paper-reported, not independently reproduced here.

## Method

The notebook uses:

```text
total loss = classification loss + λ × attention loss
attention loss = 1 − AMIL
```

It evaluates AUROC, accuracy, sensitivity, specificity, F1, ECE, AMIL, and IoU. The notebook requires external datasets, pretrained weights, and a GPU-capable environment.
