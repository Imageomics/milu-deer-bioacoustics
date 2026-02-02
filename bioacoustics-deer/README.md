# Wilds Deer Bioacoustics

A lightweight audio detection pipeline for identifying target wildlife vocalizations (e.g., deer) in field recordings.  
This repository provides tools to **train** a binary audio classifier and **run inference** on long recordings to automatically extract detected segments.

The system uses **YAMNet embeddings**, **PCA**, and **XGBoost with probability calibration** to produce interpretable detection scores.

## Subproject Overview

This repository contains:

- `model-training.py`  
  Trains a calibrated binary classifier from labeled audio data and exports a reusable model bundle.

- `demo.py`  
  Runs inference on long audio recordings by splitting them into short clips, scoring each clip, and saving detected positives along with a CSV summary.

- `requirements.txt`  
  A machine-readable list of pinned Python dependencies for reproducibility.

- `LICENSE.md`  
  Open-source license for this project.


## Dataset (Positives/Negatives)

This project uses a dataset that is too large to store directly on GitHub, so it is hosted on the Hugging Face Hub:

https://huggingface.co/datasets/imageomics/pere-david-deer-vocalizations

That link provides access to **both** the `positives/` and `negatives/` folders used to train the model.

After downloading, place them at the repository root like:

positives/  
negatives/


## Installation
Install the required Python packages using:

pip install -r requirements.txt

## Training the Model (`model-training.py`)

### Training data format

The training script expects the following folders at the project root:

positives/   # .wav files containing the target sound (label = 1)  
negatives/   # .wav files NOT containing the target sound (label = 0)

Notes:
- Input format: `.wav`
- Audio files may be any length
- Sample rates do not need to match (handled internally)

### Run training

python model-training.py

### Training output

The training script:
- Extracts YAMNet embeddings
- Applies feature scaling and PCA
- Trains an XGBoost classifier with class imbalance handling
- Applies probability calibration
- Learns a decision threshold from validation data

A trained model bundle (`.pkl`) is saved to an output directory (created automatically).  
This bundle includes all preprocessing steps and the calibrated classifier.

## Running Inference (`demo.py`)

### Supported input formats

- .wav
- .mp3
- .flac
- .ogg
- .m4a

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
python demo.py

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
