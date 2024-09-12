# Import necessary libraries
import os
import cv2
import dlib
import numpy as np
import pickle

# Initialize the face detector, shape predictor, and face recognizer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./haarcascades/shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('./haarcascades/dlib_face_recognition_resnet_model_v1.dat')

# Function to extract face descriptors from an image
def extract_face_descriptions(img_path):
    # Read the image
    img = cv2.imread(img_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = detector(gray)
    
    # List to store the face descriptors
    descriptions = []
    
    # Iterate through each detected face
    for face in faces:
        # Get the shape of the face
        shape = predictor(gray, face)
        # Compute the face descriptor for the face
        face_descriptor = face_recognizer.compute_face_descriptor(img, shape)
        # Convert the face descriptor to a numpy array
        face_descriptor = np.array(face_descriptor)
        # Normalize the face descriptor
        face_descriptor = face_descriptor / np.linalg.norm(face_descriptor)  
        # Append the face descriptor to the list
        descriptions.append(face_descriptor)
        
    # Return the list of face descriptors
    return descriptions

# Function to train the face recognition model
def train_recognizer(training_data_path):
    # Lists to store the labels, face descriptors, and label dictionary
    labels = []
    descriptors = []
    label_dict = {}
    label_id = 0
    
    # Iterate through each person's directory in the training data path
    for person_name in os.listdir(training_data_path):
        # Get the full path of the person's directory
        person_path = os.path.join(training_data_path, person_name)
        
        # Check if the person's directory exists
        if not os.path.isdir(person_path):
            print("Could not find training data")
            continue
            
        # Add the person's name to the label dictionary
        label_dict[label_id] = person_name
        
        # Iterate through each image in the person's directory
        for image_name in os.listdir(person_path):
            # Get the full path of the image
            image_path = os.path.join(person_path, image_name)
            # Extract the face descriptors from the image
            image_descriptors =  extract_face_descriptions(image_path)
            
            # Iterate through each face descriptor in the image
            for descriptor in image_descriptors:
                # Append the label to the labels list
                labels.append(label_id)
                # Append the face descriptor to the descriptors list
                descriptors.append(descriptor)
                
        # Increment the label id
        label_id += 1
        
    # Save the labels, descriptors, and label dictionary to a pickle file
    with open('face_descriptors.pkl', 'wb') as f:
        pickle.dump((descriptors, labels, label_dict), f)
        
# Run the train_recognizer function with the path to the training data directory
if __name__ == '__main__':
    train_recognizer('./data')