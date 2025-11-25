# --- Imports ---
import os, shutil, joblib, numpy as np, pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import tensorflow_hub as hub
import torch, torchaudio, soundfile as sf

# --- Local paths (normal use in IDE / GitHub) ---
positives_folder = "./positives"
negatives_folder  = "./negatives"
base_dir = "./pere_deer_output_xgb"

# Creating local output directories
for sub in ["embeddings","plots","models","reports","checkpoints"]:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
emb_dir, plot_dir, model_dir, report_dir, ckpt_dir = [
    os.path.join(base_dir, s) for s in ["embeddings","plots","models","reports","checkpoints"]
]

# Parameters
POOLING_MODE = "mean_std"
PCA_COMPONENTS = 256
RNG = 42

# Load positive/negative file lists
pos_files = glob(os.path.join(positives_folder, "*.wav"))
neg_files = glob(os.path.join(negatives_folder, "*.wav"))

files  = pos_files + neg_files
labels = [1]*len(pos_files) + [0]*len(neg_files)
data = pd.DataFrame({"file": files, "label": labels}).sample(frac=1, random_state=RNG).reset_index(drop=True)

# Splitting into 80-20 test/train
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
train_idx, val_idx = next(sss.split(data["file"], data["label"]))
train_df = data.iloc[train_idx].reset_index(drop=True)
val_df   = data.iloc[val_idx].reset_index(drop=True)

# Loading YAMNET for audio embeddings
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(YAMNET_URL)

# Creates one full embedding from all frame embeddings
def pool_frames(frames, mode="mean_std"):
    F=np.asarray(frames,dtype=np.float32)
    if F.ndim!=2 or F.shape[0]==0:
        return np.zeros((1024,),dtype=np.float32)
    if mode=="mean_std":
        return np.concatenate([F.mean(0),F.std(0)])
    return F.mean(0)

# Getting YAMNET embedding for audio file
def yamnet_embed(fp):
    try:
        wav,sr=sf.read(fp)
        if wav.ndim>1:
            wav=np.mean(wav,axis=1)
        if sr!=16000:
            wav=torchaudio.functional.resample(torch.tensor(wav),sr,16000).numpy()
        _,emb,_=yamnet_model(wav)
        return pool_frames(emb.numpy(),POOLING_MODE)
    except Exception:
        return None

# Getting embeddings for dataset
def build_embeddings(df, split):
    Xp=os.path.join(emb_dir,f"X_{split}.npy")
    yp=os.path.join(emb_dir,f"y_{split}.npy")
    feats,labels=[],[]

    for _,r in tqdm(df.iterrows(),total=len(df)):
        e=yamnet_embed(r.file)
        if e is None:
            continue
        feats.append(e)
        labels.append(r.label)

    X=np.array(feats,np.float32)
    y=np.array(labels,np.int64)
    np.save(Xp,X)
    np.save(yp,y)
    return X,y

# Build embeddings
Xy_tr,y_tr=build_embeddings(train_df,"train")
Xy_va,y_va=build_embeddings(val_df,"val")

# Normalization & PCA
scaler=StandardScaler().fit(Xy_tr)
X_tr_s=scaler.transform(Xy_tr)
X_va_s=scaler.transform(Xy_va)

pca=None
if PCA_COMPONENTS:
    pca=PCA(n_components=PCA_COMPONENTS,random_state=RNG).fit(X_tr_s)
    X_tr=pca.transform(X_tr_s)
    X_va=pca.transform(X_va_s)
else:
    X_tr,X_va=X_tr_s,X_va_s

# Handling class imbalance
pos,neg=int(y_tr.sum()),len(y_tr)-int(y_tr.sum())
spw=max(1.0,neg/max(1,pos))
print(f"pos={pos} neg={neg} spw≈{spw:.2f}")

# Early stop + calibration splits
X_fit,X_tmp,y_fit,y_tmp=train_test_split(X_tr,y_tr,test_size=0.25,stratify=y_tr,random_state=RNG)
X_es,X_cal,y_es,y_cal=train_test_split(X_tmp,y_tmp,test_size=0.5,stratify=y_tmp,random_state=RNG)
print(f"Train={len(X_fit)} | ES={len(X_es)} | CAL={len(X_cal)} | VAL={len(X_va)}")

# XGB Model declaration
xgb=XGBClassifier(
    n_estimators=4000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    scale_pos_weight=spw,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RNG
)

# Model training
try:
    xgb.fit(X_fit,y_fit,eval_set=[(X_es,y_es)],early_stopping_rounds=200,verbose=False)
except TypeError:
    xgb.fit(X_fit,y_fit,eval_set=[(X_es,y_es)],verbose=False)

# Platt calibration
xgb_cal=CalibratedClassifierCV(xgb,cv="prefit",method="sigmoid")
xgb_cal.fit(X_cal,y_cal)

# Model evaluation
def eval_model(mdl,Xv,yv):
    p=mdl.predict_proba(Xv)[:,1]
    ap=average_precision_score(yv,p)
    pr,rc,th=precision_recall_curve(yv,p)
    f1=(2*pr*rc)/(pr+rc+1e-12)
    idx=int(np.argmax(f1))
    thr=float(th[idx]) if idx < len(th) else 0.5
    return ap,p,thr,(pr,rc)

ap,pv,thr,(pr,rc)=eval_model(xgb_cal,X_va,y_va)
print(f"AP={ap:.4f} | BestThr={thr:.3f}")

# Save model locally
bundle={"model":xgb_cal,"threshold":thr,"scaler":scaler,"pca":pca,"meta":{"ap_val":float(ap)}}
joblib.dump(bundle,os.path.join(model_dir,"pere_deer_best.pkl"))
