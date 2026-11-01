from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.RandAugment(num_ops=2, magnitude=9), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class ChestXRayDataset(Dataset):
    def __init__(self, frame, root, is_rsna=False):
        self.frame = frame.reset_index(drop=True); self.root = Path(root); self.is_rsna = is_rsna; self.transform = transform()
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        if self.is_rsna:
            path = self.root / f"{row.patientId}.dcm"
            image = Image.fromarray(pydicom.dcmread(path).pixel_array).convert("RGB")
        else:
            path = self.root / (row["Image Index"] if "Image Index" in row else row.Path)
            image = Image.open(path).convert("RGB")
        image = self.transform(image)
        mask = torch.zeros((1,224,224), dtype=torch.float32)
        if self.is_rsna and int(row.Target) == 1:
            scale = 224 / 1024
            x1,y1 = int(max(0,row.x*scale)), int(max(0,row.y*scale))
            x2,y2 = int(min(224,(row.x+row.width)*scale)), int(min(224,(row.y+row.height)*scale))
            mask[:, y1:y2, x1:x2] = 1
        label = int(row.target if "target" in row else row.Target)
        return image, torch.tensor(label, dtype=torch.long), mask

def load_protocol(data_root):
    root=Path(data_root)
    nih=pd.read_csv(root/"nih_data/Data_Entry_2017.csv")
    pos=nih[nih["Finding Labels"].str.contains("Pneumonia")]; neg=nih[nih["Finding Labels"]=="No Finding"].sample(n=len(pos)*3,random_state=42)
    nih=pd.concat([pos,neg]).sample(frac=1,random_state=42).reset_index(drop=True); nih["target"]=nih["Finding Labels"].str.contains("Pneumonia").astype(int)
    chex=pd.read_csv(root/"chexpert_data/CheXpert-v1.0-small/train.csv"); chex=chex[chex["Frontal/Lateral"]=="Frontal"].copy(); cp=chex[chex.Pneumonia==1.0]; cn=chex[chex.Pneumonia!=1.0].sample(n=min(5000,len(chex[chex.Pneumonia!=1.0])),random_state=42); chex=pd.concat([cp,cn]).sample(frac=1,random_state=42).reset_index(drop=True); chex["target"]=(chex.Pneumonia==1.0).astype(int)
    rsna=pd.read_csv(root/"rsna_data/stage_2_train_labels.csv"); u=rsna.groupby("patientId")["Target"].first().reset_index(); keep=pd.concat([u[u.Target==1],u[u.Target==0].sample(n=7000,random_state=42)]); rsna=rsna[rsna.patientId.isin(keep.patientId)].reset_index(drop=True)
    return {"nih":(nih,root/"nih_data/images-224/images-224"),"chexpert":(chex,root/"chexpert_data"),"rsna":(rsna,root/"rsna_data/stage_2_train_images")}
