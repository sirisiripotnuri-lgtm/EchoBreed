const form = document.querySelector("#predict-form");
const imageInput = document.querySelector("#dog-image");
const uploadLabel = document.querySelector("#upload-label");
const previewFrame = document.querySelector(".preview-frame");
const previewImage = document.querySelector("#preview-image");
const statusText = document.querySelector("#status-text");
const resultTitle = document.querySelector("#result-title");
const confidenceText = document.querySelector("#confidence-text");
const submitButton = document.querySelector("#submit-button");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  uploadLabel.textContent = file.name;
  previewImage.src = URL.createObjectURL(file);
  previewFrame.classList.add("has-image");
  statusText.textContent = "Photo selected.";
  resultTitle.textContent = "Ready to classify.";
  confidenceText.textContent = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const name = formData.get("name");
  const age = formData.get("age");

  submitButton.disabled = true;
  submitButton.textContent = "Classifying...";
  statusText.textContent = "Asking the model...";
  resultTitle.textContent = `Meet ${name}.`;
  confidenceText.textContent = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    resultTitle.textContent = `Meet ${payload.name}! They are ${payload.age} years old and look like a ${payload.breed}.`;

    if (payload.model_loaded) {
      const confidence = Math.round(payload.confidence * 100);
      statusText.textContent = "Prediction complete.";
      confidenceText.textContent = `Confidence: ${confidence}%`;
    } else {
      statusText.textContent = "The app is running, but the trained model is not saved yet.";
      confidenceText.textContent = "Train the model to enable real breed predictions.";
    }
  } catch (error) {
    statusText.textContent = "Something went wrong.";
    resultTitle.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Classify Breed";
  }
});
