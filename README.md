# Computer Vision & Medical Imaging

This repository shows my progression from learning Computer Vision and CNN fundamentals to working on medical-imaging research.

I first practiced image classification and segmentation. After building these foundations, I worked on chest X-ray experiments and developed a research project on Clinical Attention Consistency Regularization (CACR).

## Structure

### `learning_CV/`

- `ANN_on_MNIST.ipynb` — handwritten-digit classification with an artificial neural network.
- `CV.ipynb` — DenseNet121 chest X-ray pneumonia classification and Oxford-IIIT Pet exploration.
- `Segmentation.ipynb` — U-Net with a ResNet34 encoder, masks, preprocessing, and DataLoaders.

These are learning exercises, not original research.

### `CACR/`

- `experiments.ipynb` — NIH ChestX-ray14, CheXpert, and RSNA experiments.
- `src/` — research source files.
- `results/` — experiment results and reports.

The project studies whether lesion-aware attention regularization can improve cross-domain chest X-ray diagnosis.

### `jhu_starter_task/`

A professor-provided starter task using MONAI and SuPreM for 3-D vertebrae segmentation from abdominal CT scans.

## Selected results

- DenseNet121 chest X-ray classification: **72.92% test accuracy** in the learning notebook.
- NIH subset: **5,412 images** — 1,353 pneumonia and 4,059 normal cases.
- CheXpert subset: **9,675 frontal images** — 4,675 pneumonia and 5,000 normal/other cases.
- RSNA subset: **13,012 unique patients** — 6,012 pneumonia and 7,000 normal patients.
- One recorded NIH-to-CheXpert baseline evaluation reports **AUROC 0.5426**.
- The CACR paper reports an AUROC change from **0.5369 to 0.6283 at λ = 0.5**. This is paper-reported and is not claimed as independently reproduced here.

## Reproducibility

The notebooks use Kaggle datasets, external medical-imaging data, pretrained models, and GPU/Colab paths. Datasets, credentials, and model weights are not included.

```bash
pip install -r requirements.txt
```

The notebooks are research records and may require path and environment updates before execution.
