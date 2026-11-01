import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
def summarize(y,p):
    return {"test_accuracy_pct":100*accuracy_score(y,p>=.5), "test_auroc":roc_auc_score(y,p) if len(set(y))==2 else float("nan")}
