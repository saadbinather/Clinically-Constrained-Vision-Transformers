import argparse, csv, json, random
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import CONFIGS
from src.data import load_protocol, ChestXRayDataset
from src.device import get_device
from src.models import build
from src.metrics import summarize

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,_ in loader:
            z=model(x.to(device)); ys.extend(y.numpy()); ps.extend(torch.softmax(z,1)[:,1].cpu().numpy())
    return summarize(np.array(ys),np.array(ps))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='.'); ap.add_argument('--out',default='runs'); ap.add_argument('--epochs',type=int); ap.add_argument('--only',nargs='*'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(exist_ok=True); device=get_device(); protocol=load_protocol(a.data_root); rows=[]
    for cfg in CONFIGS:
        if a.only and cfg.name not in a.only: continue
        seed(cfg.seed); frame,root=protocol[cfg.dataset]; is_rsna=cfg.dataset=='rsna'
        # This is the notebook's exact 80/20 split. Its validation holdout is also its reported test.
        labels=frame['Target'] if is_rsna else frame['target']; tr,te=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,stratify=labels)
        run=out/cfg.name; run.mkdir(parents=True,exist_ok=True); frame.iloc[tr].to_csv(run/'train_manifest.csv',index=False); frame.iloc[te].to_csv(run/'test_manifest.csv',index=False); cfg.save(run/'config.json')
        model=build(cfg.architecture,cfg.pretrained).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); lossfn=torch.nn.CrossEntropyLoss(); best=-1; best_epoch=0; logs=[]
        train=DataLoader(ChestXRayDataset(frame.iloc[tr],root,is_rsna),cfg.batch_size,shuffle=True,num_workers=0); val=DataLoader(ChestXRayDataset(frame.iloc[te],root,is_rsna),cfg.batch_size,shuffle=False,num_workers=0)
        for ep in range(a.epochs or cfg.epochs):
            model.train(); total=0
            captured={}
            hook=None
            if cfg.alignment:
                hook=model.blocks[-1].attn.register_forward_hook(lambda _m,_i,o: captured.update(attn=o.mean(-1)[:,1:].reshape(-1,14,14)))
            for x,y,mask in tqdm(train,desc=cfg.name+f' epoch {ep+1}',leave=False):
                opt.zero_grad(); logits=model(x.to(device)); l=lossfn(logits,y.to(device))
                if cfg.alignment and cfg.lambda_weight:
                    small=torch.nn.functional.interpolate(mask.to(device),size=(14,14),mode='nearest').squeeze(1)
                    attn=captured['attn']; amil=((attn*small).sum((1,2))/(attn.sum((1,2))+1e-6)).mean(); l=l+cfg.lambda_weight*(1-amil)
                l.backward(); opt.step(); total+=l.item()
            if hook: hook.remove()
            m=evaluate(model,val,device); logs.append({'epoch':ep+1,'train_loss':total/len(train),**m})
            if m['test_auroc']>best: best=m['test_auroc']; best_epoch=ep+1; torch.save(model.state_dict(),run/'best.pth')
        (run/'training_log.json').write_text(json.dumps(logs,indent=2)); model.load_state_dict(torch.load(run/'best.pth',map_location=device)); final=evaluate(model,val,device); rows.append({'model_name':cfg.name,'dataset':cfg.dataset,'best_validation_epoch':best_epoch,**final,'device':str(device)})
    if rows:
        with (out/'results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
