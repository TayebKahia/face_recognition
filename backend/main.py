from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import os
import shutil
import pickle
import cv2
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
import face_recognition
from imutils import paths
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import imghdr

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or restrict it to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Externalize configuration via environment variables
DATASET_FOLDER = os.getenv("DATASET_FOLDER", "dataset")
DATASET_TRAINING_FOLDER = os.getenv("DATASET_TRAINING_FOLDER", "dataset_training")
TRAINING_DATASET_PATH = os.getenv("TRAINING_DATASET_PATH", "train.pickle")
DETECTION_METHOD = os.getenv("DETECTION_METHOD", "cnn")

# Pydantic model for input validation
class ImageLabel(BaseModel):
    label: str

    @validator('label')
    def check_label(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Label cannot be empty')
        return v


# Function to load encodings from a pickle file
def load_encodings(encodings_path):
    if not os.path.exists(encodings_path):
        logger.error(f"The file {encodings_path} does not exist.")
        return {}
    
    logger.info(f"Loading encodings from {encodings_path}...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'encodings' in data:
        logger.info(f"Loaded {len(data['encodings'])} encodings.")
    elif isinstance(data, list):
        logger.info(f"Loaded {len(data)} encodings.")
    else:
        logger.warning(f"Unexpected data format in {encodings_path}.")
    return data

# Function to encode a single image
def encode_image(image_path, detection_method=DETECTION_METHOD):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}.")
        return None, None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    if len(encodings) == 0:
        logger.error(f"No faces found in image: {image_path}.")
        return None, None
    return encodings[0], boxes[0]

# Function to perform PCA dimensionality reduction on face encodings
def perform_pca(encodings, n_components=0.90):
    if len(encodings) == 0:
        raise ValueError("No encodings to perform PCA.")
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(encodings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA completed. Explained variance ratio: {explained_variance:.2f}")
    return reduced_data

# Function to apply Label Propagation to assign labels to unlabeled data
def perform_label_propagation(encodings, labels):
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        raise ValueError("LabelPropagation requires more than one unique label.")
    label_prop_model = LabelPropagation()
    label_prop_model.fit(encodings, labels)
    propagated_labels = label_prop_model.transduction_
    confidence_scores = label_prop_model.predict_proba(encodings)
    logger.info("Label Propagation transduction complete.")
    return propagated_labels, confidence_scores

# Function to update training data
import os

def update_training_data(image_path, encoding, label, dataset_training_folder, training_dataset_path):
    combined_data = load_encodings(training_dataset_path)
    if not combined_data:
        logger.error(f"No labeled encodings found in {training_dataset_path}.")
        return
    data_labeled = combined_data["encodings"]
    label_encoder_data = combined_data.get("label_encoder", {})
    
    label_encoder = LabelEncoder()
    if label_encoder_data.get("classes_"):
        label_encoder.classes_ = np.array(label_encoder_data["classes_"])
    else:
        label_encoder.fit([label])
    
    if label not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, label)
    
    new_entry = {
        "imagePath": image_path,
        "loc": encoding[1],
        "encoding": encoding[0].tolist(),
        "label": label_encoder.transform([label])[0]
    }
    data_labeled.append(new_entry)
    
    new_filename = os.path.basename(image_path)
    new_path = os.path.join(dataset_training_folder, new_filename)
    
    # Check if the file already exists and rename it if necessary
    base, ext = os.path.splitext(new_filename)
    counter = 1
    while os.path.exists(new_path):
        new_filename = f"{base}_{counter}{ext}"
        new_path = os.path.join(dataset_training_folder, new_filename)
        counter += 1
    
    shutil.move(image_path, new_path)
    new_entry["imagePath"] = new_path
    
    combined_data["encodings"] = data_labeled
    combined_data["label_encoder"] = {"classes_": label_encoder.classes_.tolist()}
    with open(training_dataset_path, "wb") as f:
        pickle.dump(combined_data, f)
    logger.info(f"Updated encodings saved to {training_dataset_path}.")


# Batch Processing for label propagation
async def process_label_propagation_batch(encodings_combined, labels_combined):
    reduced_data = perform_pca(encodings_combined)
    propagated_labels = perform_label_propagation(reduced_data, labels_combined)
    return propagated_labels
def clear_dataset_folder(dataset_folder):
    # Loop through files in the dataset folder
    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        
        # Check if the file is an image by using imghdr to identify it
        if os.path.isfile(file_path) and imghdr.what(file_path):
            os.remove(file_path)
            logger.info(f"Deleted {file_path} from the dataset folder.")
# FastAPI endpoint for uploading images and processing them
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    clear_dataset_folder(DATASET_FOLDER)
    file_location = os.path.join(DATASET_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    combined_data = load_encodings(TRAINING_DATASET_PATH)
    if not combined_data:
        return JSONResponse(status_code=400, content={"message": "No labeled encodings found in training dataset."})
    
    data_labeled = combined_data["encodings"]
    label_encoder_data = combined_data.get("label_encoder", {})
    label_encoder = LabelEncoder()
    if label_encoder_data.get("classes_"):
        label_encoder.classes_ = np.array(label_encoder_data["classes_"])
    else:
        label_encoder.fit([entry["label"] for entry in data_labeled])
    
    encoding, loc = encode_image(file_location)
    if encoding is None or loc is None:
        return JSONResponse(status_code=400, content={"message": "No faces found in the uploaded image."})
    
    encodings_combined = [entry["encoding"] for entry in data_labeled] + [encoding.tolist()]
    labels_combined = [entry["label"] for entry in data_labeled] + [-1]
    
    if background_tasks:
        background_tasks.add_task(process_label_propagation_batch, encodings_combined, labels_combined)
    
    propagated_labels, confidence_scores = await process_label_propagation_batch(encodings_combined, labels_combined)
    
    predicted_label = propagated_labels[-1]
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    confidence_score = confidence_scores[-1][predicted_label]
    
    return {"predicted_label": predicted_label_name, "confidence_score": confidence_score, "image_path": file_location}
# FastAPI endpoint for confirming label
@app.post("/confirm/")
async def confirm_label(image_path: str = Form(...), label: str = Form(...)):
    # Validate label using Pydantic
    label_data = ImageLabel(label=label)
    print(label)
    print(label_data)
    encoding, loc = encode_image(image_path)
    if encoding is None or loc is None:
        return JSONResponse(status_code=400, content={"message": "No faces found in the image."})
    
    update_training_data(image_path, (encoding, loc), label, DATASET_TRAINING_FOLDER, TRAINING_DATASET_PATH)
    
    return {"message": "Training data updated successfully."}

# Run the FastAPI app
if __name__ != "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



@app.get("/")
async def start():
    return{'hello':'world'}


# ...existing code...

@app.post('/reset_training/')
def reset_training():
    import os
    import shutil

    # Delete all files in dataset folder
    dataset_folder = 'dataset'
    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

    # Delete all images in dataset_training
    dataset_training_folder = 'dataset_training'
    for filename in os.listdir(dataset_training_folder):
        file_path = os.path.join(dataset_training_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

    # Copy all images from dataset_reserve into dataset_training
    dataset_reserve_folder = 'dataset_reserve'
    for filename in os.listdir(dataset_reserve_folder):
        src_file = os.path.join(dataset_reserve_folder, filename)
        dst_file = os.path.join(dataset_training_folder, filename)
        shutil.copy2(src_file, dst_file)

    # Run the command
    os.system("python ./create_training_encodings.py -d dataset_training -o train.pickle -m cnn")

    return 'Training reset successful', 200
