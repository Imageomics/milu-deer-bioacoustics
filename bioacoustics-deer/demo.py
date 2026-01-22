#Change to ensure the input folder contains the input audio .wav files
input_folder  = "input_folder_ex"
#Create name of output folder that will contain the detected clips
output_folder = "output_folder_ex"
#Provide path to the .pkl file with the ai model
bundle_path   = "detection/models/pere_deer_best.pkl"

import os, csv, math, joblib, numpy as np
import soundfile as sf
import torch, torchaudio
import tensorflow_hub as hub
from tqdm import tqdm

#Creating output folder
os.makedirs(output_folder, exist_ok=True)
#Extracting features from the ai .pkl file
bundle = joblib.load(bundle_path)
model  = bundle["model"]
scaler = bundle["scaler"]
pca    = bundle["pca"]
thr    = float(bundle["threshold"])
print("Loaded bundle keys:", list(bundle.keys()))
print(f"Threshold from bundle: {thr:.3f}")

#Loading YAMNet
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

#Feature pooling
def pool_frames(frames: np.ndarray, mode: str = "mean_std") -> np.ndarray:
    F = np.asarray(frames, dtype=np.float32)
    if F.ndim != 2 or F.shape[0] == 0:
        return np.zeros((1024,), dtype=np.float32)
    if mode == "mean":
        return F.mean(axis=0)
    if mode == "mean_std":
        return np.concatenate([F.mean(axis=0), F.std(axis=0)])
    return F.mean(axis=0)

def embed_segment_2048(wav_16k: np.ndarray) -> np.ndarray:

    _, emb, _ = yamnet(wav_16k)
    e = pool_frames(emb.numpy(), mode="mean_std")
    return e.reshape(1, -1)

def transform_for_model(e2048: np.ndarray) -> np.ndarray:

    x = scaler.transform(e2048)
    if pca is not None:
        x = pca.transform(x)
    return x

def predict_proba_segment(wav_16k: np.ndarray) -> float:

    e = embed_segment_2048(wav_16k)
    x = transform_for_model(e)
    return float(model.predict_proba(x)[0, 1])

#Splitting input into 3 sec clips
target_sr = 16000
clip_sec  = 3.0
hop_sec   = 3.0
clip_samples = int(clip_sec * target_sr)
hop_samples  = int(hop_sec  * target_sr)

#Creating csv file to hold final data
csv_path = os.path.join(output_folder, "detections.csv")

with open(csv_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["source_file", "clip_index", "start_sec", "end_sec", "probability", "label"])

    # Iterate over input files
    for fname in tqdm(sorted(os.listdir(input_folder))):
        if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue

        in_path = os.path.join(input_folder, fname)

        try:
            wav, sr = sf.read(in_path, always_2d=False)
        except Exception as e:
            print(f" Could not read {fname}: {e}")
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        if sr != target_sr:
            try:
                wav = torchaudio.functional.resample(
                    torch.tensor(wav, dtype=torch.float32), sr, target_sr
                ).numpy()
            except Exception as e:
                print(f"Resample failed for {fname}: {e}")
                continue

        n = len(wav)
        if n < clip_samples:
            continue

        #3 sec sliding windows throughout all input audio files
        i = 0
        start = 0
        while start + clip_samples <= n:
            end = start + clip_samples
            segment = wav[start:end]  # 3 sec clips

            prob = predict_proba_segment(segment)
            label = int(prob >= thr)

            # Log all predictions
            writer.writerow([fname, i, start / target_sr, end / target_sr, prob, label])

            # Save positive clips
            if label == 1:
                base = os.path.splitext(fname)[0]
                out_name = f"{base}_clip{i:04d}_p{prob:.3f}.wav"
                out_path = os.path.join(output_folder, out_name)
                sf.write(out_path, segment, target_sr)

            # Advance to next window
            i += 1
            start += hop_samples

print(f" Positive clips saved to {output_folder}")
print(f" CSV log saved to {csv_path}")

