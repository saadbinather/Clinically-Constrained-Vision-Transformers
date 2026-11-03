"""Run only the six notebook cross-domain NIH/CheXpert AUROC experiments."""
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import load_protocol, ChestXRayDataset
from src.device import get_device
from src.models import build


def split(frame, label_col):
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=42,
        stratify=frame[label_col]
    )
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


def auroc(model, loader, device):
    model.eval(); labels=[]; probs=[]
    with torch.no_grad():
        for images, y, _ in loader:
            p = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            labels.extend(y.numpy()); probs.extend(p.cpu().numpy())
    return roc_auc_score(np.asarray(labels), np.asarray(probs))


def main():
    root = Path('.')
    protocol = load_protocol(root)
    device = get_device()
    datasets = {}
    for name in ('nih', 'chexpert'):
        frame, image_root = protocol[name]
        label_col = 'target'
        train, test = split(frame, label_col)
        datasets[name] = (train, test, image_root)

    experiments = [
        ('resnet50', 'nih', 'chexpert', 'runs/nih_resnet50/best.pth'),
        ('resnet50', 'chexpert', 'nih', 'runs/chexpert_resnet50/best.pth'),
        ('vit_base_patch16_224', 'nih', 'chexpert', 'runs/nih_vit_base/best.pth'),
        ('vit_base_patch16_224', 'chexpert', 'nih', 'runs/chexpert_vit_base/best.pth'),
        ('densenet121', 'nih', 'chexpert', 'runs/nih_densenet121/best.pth'),
        ('densenet121', 'chexpert', 'nih', 'runs/chexpert_densenet121/best.pth'),
    ]
    rows=[]
    for architecture, source, target, checkpoint in experiments:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f'Missing required checkpoint: {checkpoint}')
        model = build(architecture, pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        _, target_test, target_root = datasets[target]
        loader = DataLoader(ChestXRayDataset(target_test, target_root), batch_size=32, shuffle=False, num_workers=0)
        rows.append({
            'model': architecture,
            'trained_on': source,
            'tested_on': target,
            'checkpoint': str(checkpoint),
            'test_split': 'notebook_seed42_stratified_20pct',
            'auroc': auroc(model, loader, device),
        })
        del model
        if device.type == 'mps': torch.mps.empty_cache()
    with open('cross_domain_results.csv', 'w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for row in rows: print(f"{row['model']} {row['trained_on']} -> {row['tested_on']}: AUROC={row['auroc']:.6f}")


if __name__ == '__main__': main()
