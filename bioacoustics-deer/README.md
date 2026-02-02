# Wilds Deer Bioacoustics

A lightweight audio detection pipeline for identifying target wildlife vocalizations (e.g., deer) in field recordings.  
This repository provides tools to **train** a binary audio classifier and **run inference** on long recordings to automatically extract detected segments.

The system uses **YAMNet embeddings**, **PCA**, and **XGBoost with probability calibration** to produce interpretable detection scores.

## Subproject Overview

This directory contains:

- `model-training.py`  
  Trains a calibrated binary classifier from labeled audio data and exports a reusable model bundle.

- `demo.py`  
  Runs inference on long audio recordings by splitting them into short clips, scoring each clip, and saving detected positives along with a CSV summary.

- `requirements.txt`  
  A machine-readable list of pinned Python dependencies for reproducibility.

## Training Dataset (Positives/Negatives)

This project uses the [imageomics/pere-david-deer-vocalizations dataset](https://huggingface.co/datasets/imageomics/pere-david-deer-vocalizations), published on Hugging Face.

To work with this codebase, download **both** the `positives/` and `negatives/` folders into the repository root.

## Installation
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Training the Model (`model-training.py`)

### Training data format

The training script expects the following folders at the project root:

```
positives/   # .wav files containing the target sound (label = 1)  
negatives/   # .wav files NOT containing the target sound (label = 0)
```

Notes:
- Input format: `.wav`
- Audio files may be any length
- Sample rates do not need to match (handled internally)

### Run training

```bash
python model-training.py
```

### Training output

The training script:
- Extracts YAMNet embeddings
- Applies feature scaling and PCA
- Trains an XGBoost classifier with class imbalance handling
- Applies probability calibration
- Learns a decision threshold from validation data

A trained model bundle (`.pkl`) is saved to the output directory specified in `model-training.py` (default: `./pere_deer_output_xgb/`). This folder (and its subfolders) is created automatically if it does not exist. (see [configure and run](#configure-and-run). The bundle includes all preprocessing steps and the calibrated classifier.

### Training output directory

`model-training.py` saves all training artifacts to the following folder (created automatically):

```
pere_deer_output_xgb/
├── embeddings/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   └── y_val.npy
├── models/
│   └── pere_deer_best.pkl
├── plots/
├── reports/
└── checkpoints/

```

## Running Inference (`demo.py`)

### Supported input formats

- `.wav`
- `.mp3`
- `.flac`
- `.ogg`
- `.m4a`

### How inference works

1. Loads each recording  
2. Resamples audio to 16 kHz if needed  
3. Splits audio into fixed-length clips (3 seconds)  
4. Scores each clip independently  
5. Saves detected clips and a CSV report  

### Configure and run

Edit the following paths at the top of `demo.py`:
- `input_folder` — folder containing audio recordings  
- `output_folder` — where detections will be saved  
- `bundle_path` — path to the trained `.pkl` model bundle  

Then run:
```bash
python demo.py
```

### Output structure

```
output_folder/  
├── detections.csv  
├── recording1_clip0003_p0.912.wav  
├── recording2_clip0017_p0.876.wav  
└── ...  
```

The `detections.csv` file includes clip start/end times, probabilities, and source filenames.


## Reproducibility

- All Python dependencies are pinned in `requirements.txt`
- The trained model bundle includes preprocessing, PCA, calibration, and thresholding
- Designed to be portable across systems with minimal setup


## External Tools and Resources

This project makes use of the following tools and libraries:

- YAMNet (TensorFlow Hub) — pretrained audio embedding model  
- XGBoost — gradient-boosted decision trees for classification  
- scikit-learn — preprocessing, PCA, and calibration utilities  
- PyTorch / torchaudio — audio loading and resampling  


## License

This project is licensed under the terms specified in the `LICENSE.md` file at the root of the repository.

## Author

**Varun Viswapriyan**  
GitHub: https://github.com/varunviswapriyan
