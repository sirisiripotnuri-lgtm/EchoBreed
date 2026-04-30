from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.model_utils import DogBreedClassifier

app = FastAPI(title="Dog Breed Classifier")
classifier = DogBreedClassifier()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
INDEX_PATH = Path("app/templates/index.html")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True, "model_loaded": classifier.is_ready}


@app.post("/predict")
async def predict(
    name: str = Form(...),
    age: str = Form(...),
    image: UploadFile = File(...),
) -> Dict[str, Union[str, float, bool]]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    breed, confidence = classifier.predict(pil_image)

    return {
        "name": name.strip(),
        "age": age.strip(),
        "breed": breed,
        "confidence": round(confidence, 4),
        "model_loaded": classifier.is_ready,
    }
