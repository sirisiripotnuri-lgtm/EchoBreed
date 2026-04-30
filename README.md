# Dog Breed Classifier with Metadata

## News

- A trained MobileNetV2 transfer-learning model is included in `models/dog_breed_model.keras`.
- The Stanford Dogs dataset archives were prepared into 120 breed folders with 20,580 cropped dog images.
- The prediction pipeline now handles common image-orientation issues, including rotated uploads.
- The web app collects dog name, age, and image, then returns a breed prediction with confidence.

## Introduction

This project is a full-stack machine learning application for dog breed classification. It uses a convolutional neural network built with TensorFlow/Keras and transfer learning from MobileNetV2. The model predicts one of 120 breeds from the Stanford Dogs dataset.

The application also includes a FastAPI backend and a simple browser interface. Users can enter a dog's name and age, upload a photo, and receive a friendly result such as:

```text
Meet Shiro! They are 3 years old and look like a Pomeranian.
```

## Main Results

The included model was trained for 1 epoch as a fast baseline.

| Dataset | Breeds | Images | Train Split | Validation Split | Validation Accuracy |
|---|---:|---:|---:|---:|---:|
| Stanford Dogs | 120 | 20,580 | 16,464 | 4,116 | 81.71% |

### Tested Examples

| Image Type | Expected Breed | Predicted Breed | Confidence |
|---|---|---|---:|
| Pomeranian image | Pomeranian | Pomeranian | 77.30% |
| Rotated husky image | Siberian husky | Siberian husky | 62.11% |

Note:

- Accuracy can improve by training for more epochs.
- Some breeds are visually similar, such as Siberian husky, Eskimo dog, and malamute.
- Real-world images may be harder than validation images because of pose, lighting, cropping, and rotation.

## Environment

The project was developed on macOS using Python 3.9.

Required packages are listed in `requirements.txt`:

- FastAPI
- Uvicorn
- TensorFlow
- Pillow
- NumPy
- python-multipart

## Quick Start

### Installation

Open the project folder in VS Code:

```text
/Users/sirisha/Documents/New project
```

Open a new VS Code terminal and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `.venv` already exists, use:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Run on Localhost

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001
```

Keep the terminal open while using the app. If the terminal is closed, the localhost site will stop running.

## Project Structure

The project should look like this:

```text
dog-breed-classifier/
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── model_utils.py
│   ├── static
│   │   ├── app.js
│   │   └── styles.css
│   └── templates
│       └── index.html
├── data
│   ├── processed-dogs
│   │   ├── Chihuahua
│   │   ├── Pomeranian
│   │   ├── Siberian husky
│   │   └── ...
│   └── stanford-dogs
│       ├── Annotation
│       └── Images
├── models
│   ├── breed_labels.json
│   └── dog_breed_model.keras
├── training
│   ├── prepare_stanford_dogs.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

Note:

- The runnable zip includes the trained model and code.
- The raw `data/` folder is large and is ignored by Git.
- The virtual environment `.venv/` is also ignored by Git.

## Data Preparation

The Stanford Dogs archives are expected in this format:

```text
data/stanford-dogs/
├── Annotation
│   ├── n02085620-Chihuahua
│   └── ...
└── Images
    ├── n02085620-Chihuahua
    └── ...
```

Prepare cropped breed folders:

```bash
python training/prepare_stanford_dogs.py
```

This creates:

```text
data/processed-dogs/
├── Chihuahua
├── Japanese spaniel
├── Maltese dog
├── Pomeranian
├── Siberian husky
└── ...
```

The preparation script:

- Reads Stanford Dogs annotation XML files.
- Crops each image using the dog bounding box.
- Saves images into clean breed-name folders.

## Training

Train the model:

```bash
python training/train.py --epochs 8
```

For a quick baseline:

```bash
python training/train.py --epochs 1
```

Training saves:

```text
models/dog_breed_model.keras
models/breed_labels.json
```

The model uses:

- MobileNetV2 pretrained on ImageNet.
- Frozen convolutional base.
- Global average pooling.
- Dropout.
- Dense softmax output layer for 120 breeds.

## Validating the API

Check if the server is running:

```bash
curl http://127.0.0.1:8001/health
```

Expected response:

```json
{
  "ok": true,
  "model_loaded": true
}
```

## API

### Predict

Endpoint:

```text
POST /predict
```

Form fields:

| Field | Type | Description |
|---|---|---|
| `name` | text | Dog name |
| `age` | text/number | Dog age |
| `image` | file | Dog photo |

Example response:

```json
{
  "name": "Shiro",
  "age": "3",
  "breed": "Pomeranian",
  "confidence": 0.773,
  "model_loaded": true
}
```

## Running in VS Code

1. Open VS Code.
2. Select `File > Open Folder`.
3. Choose:

```text
/Users/sirisha/Documents/EchoBreed
```

4. Open `Terminal > New Terminal`.
5. Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "uvicorn[standard]"
uvicorn app.main:app --host 127.0.0.1 --port 8001

```
6. Open:

```browser
http://127.0.0.1:8001
```
If .venv already exists, you can skip the first command and just run:
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8001
```


## Troubleshooting

### Site is not loading

The server is probably not running. Start it again:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### Model not trained yet

Make sure these files exist:

```text
models/dog_breed_model.keras
models/breed_labels.json
```

### Wrong breed prediction

Try:

- A clearer dog image.
- A photo where the dog is upright.
- A photo with the full dog visible.
- Training for more epochs.

The app includes a small orientation check for weak predictions, but more training will improve reliability.

## Other Implementations

This project can be extended with:

- React frontend.
- Streamlit interface.
- Flask backend.
- TensorFlow Lite mobile model.
- Grad-CAM visualization.
- Fine-tuning MobileNetV2 or replacing it with EfficientNet.

## Citation / Dataset

If you use the Stanford Dogs dataset, cite the original dataset source:

```text
Stanford Dogs Dataset
http://vision.stanford.edu/aditya86/ImageNetDogs/
```

This project uses TensorFlow/Keras MobileNetV2 transfer learning for educational purposes.
