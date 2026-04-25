# VocalID: A Lightweight Voice Authentication Toolkit

[![PyPI Downloads](https://img.shields.io/pypi/dm/vocalid?label=PyPI%20Downloads)](https://pypi.org/project/vocalid/)

VocalID is a practical and lightweight voice authentication library built around ECAPA-TDNN speaker embeddings and a simple classification layer. It lets you train your own voice model, evaluate its performance, and verify identities from recorded or live audio to use voice as a biometric to unlock apps, applications and for privacy. The goal is to make voice verification simple to run, easy to extend, and stable across devices.

---

## Features

* ECAPA-TDNN speaker embeddings (`speechbrain/spkrec-ecapa-voxceleb`)
* Easy training workflow for positive (owner) and negative samples
* Evaluation with accuracy and a full classification report
* File-based verification and optional live microphone verification
* Clean CLI for training, testing and verification
* Modular, readable codebase
* Simple model storage using pickle
* Test suite included
* Simple FastAPI server for verification

---

## How It Works

**1. Audio Processing**
Audio is loaded or recorded, resampled to the target rate, converted to mono and padded to a minimum length.

**2. Embedding Extraction**
We extract fixed-dimensional embeddings using ECAPA-TDNN. These embeddings capture speaker-specific characteristics.

**3. Training**
A logistic regression classifier is trained on positive and negative embeddings.

**4. Verification**
When verifying a sample:

1. Extract the embedding
2. Run it through the model
3. Get a probability score
4. Compare with the threshold from `config.py`

---

## Package Structure

```
VocalID/
│
├── voice_verifier/
│   ├── trainer.py        # Training, evaluation, saving, loading
│   ├── verifier.py       # Verification from file or tensor
│   ├── embeddings.py     # ECAPA-TDNN embedding extractor
│   ├── audio_utils.py    # Audio loading / microphone recording
│   ├── config.py         # Threshold, sample rate, model config
│   ├── model_store.py    # Pickle storage helpers
│   ├── cli.py            # CLI interface
│
├── tests/                # Pytest suite
├── examples/             # Example scripts
├── requirements.txt
├── api/app.py            # Optional API server example
└── README.md
```

---

## Installation

> For **windows** users

```bash
pip install vocalid
```

Or install from source:

```bash
git clone https://github.com/Khubaib8281/VocalID.git
cd VocalID
pip install -e .
```

> For **Linux** Users

Also install;

```ubuntu
apt-get install -y libportaudio2
```

Since, `soundevice` relies on `libportaudio2`

---

## Dataset Layout

Your dataset should be organized as:

```
dataset/
│
├── my_voice/            # Positive samples (your voice)
│   sample1.wav
│   sample2.wav
│   ...
│
└── other_voices/        # Negative samples (others)
    voice1.wav
    voice2.wav
    ...
```

Each sample should ideally be 4–6 seconds with varied tone, distance, and background conditions.

---

## Example Usage (Python)

### Train

```python
from vocalid.trainer import VoiceTrainer
import glob

pos_files = glob.glob("dataset/my_voice/*.wav")
neg_files = glob.glob("dataset/other_voices/*.wav")

trainer = VoiceTrainer()
trainer.train(pos_files, neg_files, save_path="my_voice_model.pkl")
```

### Evaluate

```python
trainer.load("my_voice_model.pkl")

test_pos = glob.glob("dataset/my_voice_test/*.wav")
test_neg = glob.glob("dataset/other_voices_test/*.wav")

metrics = trainer.evaluate(test_pos, test_neg)
print("Accuracy:", metrics["accuracy"])
print(metrics["report"])
```

### Verify a file

```python
from vocalid.verifier import VoiceVerifier

verifier = VoiceVerifier("my_voice_model.pkl")
ok, score = verifier.verify_file("verify_samples/unknown.wav")

print(ok, score)
```

### Verify live audio

```python
ok, score = verifier.verify_live(audio_tensor)
print(ok, score)
```

Live recording only works on systems with a real microphone. It will not run in cloud notebooks.

---

## CLI Usage

Train:

```bash
vocalid train --positive my_voice --negative others --output model.pkl
```

Evaluate:

```bash
vocalid evaluate --model model.pkl --positive my_voice --negative others
```

Verify a file:

```bash
vocalid verify sample.wav --model model.pkl
```

Live verification:

```bash
vocalid live --model model.pkl --seconds 4
```

---

## Use Cases

* Personal voice-unlock systems for apps, applications and systems
* Lightweight speaker verification
* Research in speaker embeddings
* Prototyping identity checks
* Classroom or research demonstrations
* Testing spoofing and adversarial audio

---

## Why This Matters

VocalID helps developers learn how practical speaker verification works without dealing with heavy frameworks. The library focuses on transparency, modularity and simplicity:

* Clear separation of embedding extraction and classification
* Easy to swap in a different classifier
* Works on CPU
* No special hardware needed for training

---

## Contributing

Pull requests are welcome. To run tests:

```bash
pytest -v
```

Feel free to open issues for bugs, improvement ideas, or feature requests.

---

## Author

**Muhammad Khubaib Ahmad**

AI/ML Research Engineer | Speech and Language Intelligence

- [**Portfolio:**](https://huggingface.co/spaces/Khubaib01/KhubaibAhmad_Portfolio)
- **Email** [khubaib0.1ai@gmail.com](mailto:khubaib0.1ai@gmail.com)
- [**GitHub**](https://github.com/Khubaib8281)
- [**LinkedIn**](https://www.linkedin.com/in/muhammad-khubaib-ahmad-)
- [**Kaggle**](https://kaggle.com/muhammadkhubaibahmad)
- [**HuggingFace**](https://huggingface.co/Khubaib01)

---

## License

MIT License
