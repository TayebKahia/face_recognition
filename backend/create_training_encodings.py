# create_training_encodings.py
"""
python .\create_training_encodings.py -d dataset_training -o train.pickle -m cnn
"""
import os
import pickle
import argparse
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import face_recognition

def encode_faces(dataset_path, output_path, detection_method="cnn"):
    """
    Encodes faces from a labeled dataset and serializes the encodings along with the LabelEncoder
    into a single pickle file.

    Parameters:
        dataset_path (str): Path to the dataset directory containing labeled images.
        output_path (str): Path to save the serialized encodings and LabelEncoder.
        detection_method (str): Face detection model to use ('cnn' or 'hog').
    """
    print("[INFO] Quantifying faces...")
    # List all image files in the dataset directory
    image_paths = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    data = []

    # Loop over the image paths
    for i, image_path in enumerate(image_paths):
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}: {image_path}")

        # Extract the label from the filename (e.g., ismail.jpg -> ismail)
        filename = os.path.basename(image_path)
        label = os.path.splitext(filename)[0]

        # Load the image and convert it from BGR to RGB
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not read image: {image_path}. Skipping.")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        boxes = face_recognition.face_locations(rgb, model=detection_method)

        if len(boxes) == 0:
            print(f"[WARNING] No faces found in image: {image_path}. Skipping.")
            continue

        # Compute facial encodings for each face detected
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Associate each encoding with its label and location
        for (box, encoding) in zip(boxes, encodings):
            data.append({
                "imagePath": image_path,
                "loc": box,  # Add the face location
                "encoding": encoding.tolist(),  # Convert numpy array to list for serialization
                "label": label  # Assign label directly here
            })

    if not data:
        print("[ERROR] No face encodings found. Exiting.")
        return

    # Initialize and fit the LabelEncoder
    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    labels = [entry["label"] for entry in data]  # Extract labels corresponding to each encoding
    encoded_labels = le.fit_transform(labels)

    # Assign encoded labels to the data
    for i in range(len(data)):
        data[i]["label"] = int(encoded_labels[i])  # Ensure label is an integer

    # Create a combined dictionary
    combined_data = {
        "encodings": data,
        "label_encoder": {
            "classes_": le.classes_.tolist()  # Convert numpy array to list for serialization
        }
    }

    # Serialize the combined data to a single pickle file
    print("[INFO] Serializing encodings and LabelEncoder into one file...")
    with open(output_path, "wb") as f:
        pickle.dump(combined_data, f)
    print(f"[INFO] Combined data saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Face Encoding Script")
    parser.add_argument('-d', '--dataset', required=True, help='Path to input directory of labeled faces + images')
    parser.add_argument('-o', '--output', required=False, default='train.pickle', help='Path to output pickle file')
    parser.add_argument('-m', '--method', type=str, default='cnn', help='Face detection model to use: either `cnn` or `hog`')

    args = parser.parse_args()

    encode_faces(dataset_path=args.dataset, output_path=args.output, detection_method=args.method)

if __name__ == "__main__":
    main()