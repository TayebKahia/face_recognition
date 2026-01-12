# face_recognition (FastAPI backend)

**Simple face recognition backend** using `face_recognition` (dlib-based encodings), OpenCV, scikit-learn, and FastAPI.  
This service provides tools to create/serialize face encodings from a labeled dataset, run a FastAPI server for inference, confirm labels, and reset training data.

---

## Features
- Compute facial encodings from labeled images and serialize them into a single `train.pickle` file (encodings + label encoder). :contentReference[oaicite:9]{index=9}  
- FastAPI endpoints for:
  - Uploading an image and predicting the person using PCA + Label Propagation. :contentReference[oaicite:10]{index=10}
  - Confirming a predicted label and updating training data. :contentReference[oaicite:11]{index=11}
  - Resetting training (copies from `dataset_reserve` to `dataset_training` and regenerates encodings). :contentReference[oaicite:12]{index=12}

---

## Repo structure (relevant files)
